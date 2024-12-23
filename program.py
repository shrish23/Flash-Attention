import torch

import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr
):
    # range of values handled by this stage
    if STAGE == 1:
        # From 0 to the left of the diagonal
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        # Used only for the block in which there is transition between non-masked and masked keys
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)# telling triton this is a multiple of something
    else:
        #Only used for non-causal attention: no need to mask out anything
        lo, hi = 0, SEQ_LEN

    K_block_ptr = tl.advance(K_block_ptr, (0,lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo,0))

    #loop over the key and value and update accumulator
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        # Just let the compiler know that start_n is a multiple of BLOCK_N, so the compiler can do optimizations
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)# Telling the triton compiler this information helps improve its pipeline algorithm for the "for loop".

        # ---- compute qk -----#
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)# We had already inverted the K_block in the make_block_ptr method

        if STAGE == 2:
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))# Computing the maximum of each row
            QK_block -= m_ij[:, None]
        else:
            # Compute the maximum value of qk or keep the old max value
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        # Compute the exponential of each dot product, so now we are computing exp(qk_ij - m_ij)
        P_block = tl.math.exp(QK_block)

        # Compute the sum of the rows of the attention scores
        l_ij = tl.sum(P_block, 1)

        #Correction factor
        alpha = tl.math.exp(m_i - m_ij)# m_i is the maximum in the previous iteration

        # Apply the correction factor to the previous l_i and add the new l_ij
        l_i = l_i * alpha + l_ij

        V_block = tl.load(V_block_ptr)

        P_block = P_block.to(tl.float16)

        # This computes the following: O_new = P x V + o_old * alpha
        O_block = O_block * alpha[:,None]
        O_block = tl.dot(P_block, V_block, O_block)# This is the same as doing O_block += P_block @ V_block

        m_i = m_ij# new maximum

        #Move to the next block of K and V
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))

    return O_block, l_i, m_i

@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in ([3,4,7])
        for num_warps in [2,4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)

@triton.jit
def _attn_fwd(
    Q, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale,
    M, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    O, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    # We didn't pass the BLOCK_SIZE_Q and BLOCK_SIZE_KV when calling the method because they will be passed when we apply the Auto Tuning Decorator.
    BLOCK_SIZE_Q: tl.constexpr,# NUmber of queries to group together to make a block matrix
    BLOCK_SIZE_KV: tl.constexpr,# number of keys and values we need to group together to make a block matrix
    STAGE: tl.constexpr
):

    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)
    # This indicate which block in the sequence length to process
    block_index_q = tl.program_id(0)# Tells the index of the block of queries we are going to work with
    # This indicates which head and batch to process. Each program is associated with a single head of a single batch
    index_batch_head = tl.program_id(1)# This is the product of BATCH_SIZE * NUM_HEADS: means that we will have as many programs on the axis 1 as BATCH_SIZE * NUM_HEADS. 
    #This indicate which batch this program is associated with  (each batch has NUM_HEADS heads)
    index_batch = index_batch_head // NUM_HEADS
    # This indicate the position of the head in the batch
    index_head = index_batch_head % NUM_HEADS

    # Now the Q,K,V we have here are pointers to the starting element of the tensor. But now we need to access the index_batch and the index_head in these tensors.
    #For that we use the stride to move to that batch and head.
    #We calculate the memory offset for accessing the Q,K,V tensors. (This is important for parallel computation in Triton)
    qvk_offset = (
        index_batch.to(tl.int64) * stride_Q_batch
        + index_head.to(tl.int64) * stride_Q_head
    )

    #Triton allows us to index elements inside of the tensor without having to deal with the complex calculation of the index.
    # This is called the mask_block_ptr function.
    # We can treat this the same as tensor in pytorch, we will be able to increase one index in any dimension without computing the strides
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,# Q[index_batch, index_head, block_index_q * BLOCK_SIZE_Q, :]
        shape=(SEQ_LEN, HEAD_DIM),# shape of the tensor
        strides=(stride_Q_seq, stride_Q_dim),# strides for the sequence and the head dimension.
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1,0)
    )

    V_block_ptr = tl.make_block_ptr(# V[index_batch, index_head, :, :]
        base=V + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1,0)
    )

    K_block_ptr = tl.make_block_ptr(# K[index_batch, index_head, :, :]
        base=K + qvk_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(stride_K_dim, stride_K_seq),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0,1)
    )

    O_block_ptr = tl.make_block_ptr(# O[index_batch, index_head, block_index_q * BLOCK_SIZE_Q, :]
        base=O + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1,0)
    )

    # Now, we define an offset that will load BLOCK_SIZE_Q number of queries from the big tensor that contains all the queries
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)#: the offsets for the tokens in the Q to process.

    #offs_kv: the offsets for the tokens in K and V sequence to process
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)# In this we are not skipping anything as we are pointing to the first block of key and value, as in case of queries the program will only work with a single block of queries.

    # m_i: the running maximum. We have one for each query
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("-inf")

    # l_i: the running sum. We have one for each query (as we sum the attention scores by rows)
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0# The +1 is to make the log stable because later we will use l_i to compute logsumexp

    # acc: the accumulator for the output, which is a group of rows of the O matrix.
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)# This is the output block at index_batch and index_head

    # load the blocks of Q: it will stay in SRAM throughout
    Q_block = tl.load(Q_block_ptr)# Shifting the 

    #stage: 3 if causal else 1
    if STAGE == 1 or STAGE == 3:
        # This step runs for non-causal attention or for the blocks to the left of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN
        )

    if STAGE == 3:
        # This step runs for non-causal attention or for the blocks to the left of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN
        )

    m_i += tl.math.log(
        l_i
    )# This is needed to compute the logsumexp for the backward pass

    O_block = O_block / l_i[:, None]# Applying thr normalization to the output
    # Now to store the m_i value in order to access it during the backward pass
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q# skipping the batches and the sequence lengths to find the right head to store the information
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))

@triton.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D,# (BATCH_SIZE, NUM_HEADS, SEQ_LEN)
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr
):
    block_index_q = tl.program_id(0)# getting the index of block of Q we need to work with
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)# skipping some elements that have been processed by other programs and moving to the vector this program is going to work with
    index_batch_head = tl.program_id(1)# getting the batch index and the head we need to work with in the block
    offs_dim = tl.arange(0, HEAD_DIM)

    # Load a single block of BLOCK_SIZE_Q rows of O (Since the shape of Q and O are same)
    O_block = tl.load(# load is given an array of pointers to be loaded
        O# shape: (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM
        + offs_dim[None, :]
    )# (BATCH_SIZE_Q, HEAD_DIM)

    dO_block = tl.load(
        dO
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM
        + offs_dim[None, :]
    ).to(tl.float32)

    # Computing the D block
    D_block = tl.sum(dO_block * O_block, axis=1)
    # store the D block
    D_block_ptrs = D + index_batch_head * SEQ_LEN + offs_q
    tl.store(D_block_ptrs, D_block)# Storing the D block in the memory pointed by the D_block_ptrs

@triton.jit
def _attn_bwd_dk_dv(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr
):
    #defining the offset to enter the right batch and head
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
        tl.int64
    )
    # This is the offset that allows us to select the right sequence given the batch and head
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # Make sure the pointers are in the right place w.r.t batch and head
    # The reason we don't access the blocks through make_block_ptr is because we need to use the range of offsets to apply the masking
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    # Make sure the pointers are in the right place w.r.t batch, head and sequence
    M += offset_batch_head_seq
    D += offset_batch_head_seq

    # Now, we need to load a 2D tensor, so we need to define the range's in the second dimension of each k and V vector that we need to load
    #load scales
    offs_dim = tl.arange(0, HEAD_DIM)

    index_block_kv = tl.program_id(0)
    start_kv = index_block_kv * BLOCK_KV# skip some kv blocks which are being managed by other programs

    offs_kv = start_kv + tl.arange(0, BLOCK_KV)

    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop
    K_block = tl.load(
        K + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )# shape: (BLOCK_KV, HEAD_DIM)
    V_block = tl.load(
        V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )# shape: (BLOCK_KV, HEAD_DIM)

    offs_q = tl.arange(0, BLOCK_Q)

    #Now, we need to access transposed Q vectors (Because we need to compute the Pij which is the softmax of Q.KT) and the O vectors as we are going to iterate on them as well
    # We access the Q as a transposed array, so that's why we treat offs_q as a column vector and offs_dim as a row vector
    # This is equivalent to doing:
    # q_ptrs = Q + offs_q[:,None] * stride_seq + offs_dim[None,:] * stride_dim
    # qT_ptrs = tl.trans(q_ptrs)
    # We point to the first BLOCK_Q rows of Q for both the qT and dO pointers, inside the for loop we will move forward by BLOCK_Q rows at each iteration
    qT_ptrs = Q + offs_q[None,:] * stride_seq + offs_dim[:,None] * stride_dim# Go to the query starting point which is already pointing to the right batch and head for which this program should work with
    # and select a 2D vector along rows(but since the queries are transposed these are along columns)
    dO_ptrs = dO + offs_q[:,None] * stride_seq + offs_dim[None,:] * stride_dim

    # Iterate over the sequence dimensions of the query
    curr_q = 0
    num_steps = SEQ_LEN // BLOCK_Q
    for blk_idx in range(num_steps):
        # Load a block of Q
        qT_block = tl.load(qT_ptrs)
        # Load the logsumexp values for the queries in the current block
        offs_q = curr_q + tl.arange(0, BLOCK_Q)# This is to compute the P^T on the fly so we are loading the Logsumexp(M) matrix
        m = tl.load(M + offs_q)

        # This gives us (QK^T)^T = (K^T)^T(Q^T) = K(Q^T) = S^T
        QK_T_block = softmax_scale * tl.dot(K_block, qT_block)# This is S^T
        # We apply the softmax by using the logsumexp trick
        P_T_block = tl.math.exp(QK_T_block - m[None, :])

        if STAGE == 3:
            # Autoregressive masking
            # mask is True for all values that DO NOT NEED TO BE MASKED
            mask_block = (
                offs_q[None, :] >= offs_kv[:, None]
            )# shape: (BLOCK_KV, BLOCK_Q)
            # Replace all the masked values with 0
            # In this case we do not need to mask with -Inf before applying the softmax since we already computed the normalization factors (stored in 'm')
            P_T_block = tl.where(mask_block, P_T_block, 0.0)

        dO_block = tl.load(dO_ptrs)
        # According to the formula: dV_new = dV_old + P^T x dO, where x is the matrix multiplication
        dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)

        #Delta = rowsum(O * dO) where * is the element-wise product
        Di = tl.load(D + offs_q)# We load the same number of block size

        # dP = dO x V^T, so dP^T = V x dO^T
        # Where x is the matrix multiplication
        dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)

        # We know that dS = P * (dP - delta), so dS^T = P^T * (dP^T - Delta^T)
        dS_T_block = P_T_block * (dpT_block - Di[None, :])
        dS_T_block = dS_T_block.to(tl.float16)

        # According to the formula on the paper: dK_new = dK_old + dS^T x Q
        dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))
        #Increment pointers.
        curr_q += BLOCK_Q
        qT_ptrs += BLOCK_Q * stride_seq
        dO_ptrs += BLOCK_Q * stride_seq# Because dO and dQ have the same shape

    # Write back to dV
    dV_block_ptrs = dV + offs_kv[:,None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dV_block_ptrs, dV_block)

    # Write back dK
    dK_block_ptrs = dK + offs_kv[:,None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dK_block_ptrs, dK_block)


@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr
):
    #defining the offset to enter the right batch and head
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
        tl.int64
    )
    # This is the offset that allows us to select the right sequence given the batch and head
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # Make sure the pointers are in the right place w.r.t batch and head
    # The reason we don't access the blocks through make_block_ptr is because we need to use the range of offsets to apply the masking
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    # Make sure the pointers are in the right place w.r.t batch, head and sequence
    M += offset_batch_head_seq
    D += offset_batch_head_seq

    #load scales
    offs_dim = tl.arange(0, HEAD_DIM)

    index_block_q = tl.program_id(0)# Amongst all the queries which query block we need to work with

    start_q = index_block_q * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)

    Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
    dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    dO_block = tl.load(
        dO + offs_q[:,None] * stride_seq + offs_dim[None, :] * stride_dim
    )

    M_block = tl.load(M + offs_q)
    M_block = M_block[:, None]

    offs_kv = tl.arange(0, BLOCK_KV)# a column vector

    # We access the k and V as transposed blocks
    kT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim# here it is treated as row vector
    vT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim

    Di = tl.load(D + offs_q)

    curr_kv = 0
    num_steps = SEQ_LEN // BLOCK_KV
    for blk_idx in range(num_steps):
        # Loading the first block of K transpose and V transpose in SRAM
        K_T_block = tl.load(kT_ptrs)
        V_T_block = tl.load(vT_ptrs)

        QK_block = softmax_scale * tl.dot(Q_block, K_T_block)# Query in the current query block with transposed key
        P_block = tl.math.exp(QK_block - M_block)

        if STAGE == 3:
            # Autoregressive masking
            offs_kv = curr_kv + tl.arange(0, BLOCK_KV)
            mask_block = offs_q[:, None] >= offs_kv[None, :]# Mask is true only for those elements for which query index is more than key index
            P_block = tl.where(mask_block, P_block, 0.0)

        # Compute dP and dS
        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
        dS_block = P_block * (dP_block - Di[:, None])
        dS_block = dS_block.to(tl.float16)
        # Compute dQ
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled
        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))
        # Increment pointers
        curr_kv += BLOCK_KV
        kT_ptrs += BLOCK_KV * stride_seq
        vT_ptrs += BLOCK_KV * stride_seq

    dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dQ_block_ptrs, dQ_block)

# We need to derive the torch.autograd.Function for every operation in pytorch, be it ReLU, Softmax. This gives us two functions, namely the forward and the backward pass
class TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):# ctx is just the storage area to store some values which will be needed for calculation.
        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V = V.shape[-1]

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V# Checking if the head dimension values for qeury, key and value is same.

        O = torch.empty_like(Q)# Pre-allocate the output tensor: haveing the same shape as the Query, but not the same as key and value because of Cross Attention.
        stage = 3 if causal else 1

        grid = lambda args: (
            #ceil(SEQ_LEN / BLOCK_SIZE_Q) = How many blocks of Q we have.
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),# divide queries into block of queries
            BATCH_SIZE * NUM_HEADS,#launch program along two dimensions: Which head of which batch element are we going to work with
            1,# Z dimension in the CUDA launch grid
        )# Launching the grid that tells the triton about the number of parallel processes to be launched by CUDA(through Triton)
        # In triton we want to parallelize along the batch dimension

        # Number of parallel programs: (BATCH_SIZE * NUM_HEADS * NUM_BLOCKS_Q)

        # M is the logsumexp for the backward pass, one for each query
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )

        # launching the kernel for the forward pass and defining the launch grid.
        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M,# information we need to save in the backward pass. It has the max of the row and the normalization factor calculated in the forward pass
            O=O,# Where the kernel stores the output
            stride_Q_batch=Q.stride(0),#the stride will alow us to access the tensor with the pointer only pointing to the first element of the tensor.
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage#indicates causal or non causal attention
        )

        # Saving some information for the backward pass
        ctx.save_for_backward(Q,K,V,O,M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, M = ctx.saved_tensors# Accessing the computed variables during the forward pass

        assert dO.is_contiguous()
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()
        #Variables to store the result of backward pass:
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]
        NUM_WARPS, NUM_STAGES = 4, 3
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128

        preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)# launch grid for this function, and it will be launched independently for each batch and each head
        D = torch.empty_like(M)# shape: (BATCH_SIZE, NUM_HEADS, SEQ_LEN)

        # compute all the elements Di
        _attn_bwd_preprocess[preprocess_grid](
            O=O,
            dO=dO,
            D=D,
            SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE_Q=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM
        )

        # creating a launch grid for next iteration, since we need to keep one block fixed and iterate through all the other blocks
        # We define how man programs we have that runin parallel and the block is fixed has a block size of BLOCK_SIZE_MACRO
        grid = (SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE * NUM_HEADS)

        stage = 3 if ctx.causal else 1

        # in first iteration we will fix K and V blocks and iterate through all Q blocks in size of BLOCK_SIZE_MICRO
        _attn_bwd_dk_dv[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MICRO,
            BLOCK_KV=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES
        )

        #Fix Q and iterate through all the KV block
        _attn_bwd_dq[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MICRO,
            BLOCK_KV=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES
        )

        return dQ, dK, dV, None, None


def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1 / (HEAD_DIM**0.5)# QK*t/sqrt(HEAD_DIM)
    dO = torch.randn_like(Q)# Needed for the backward pass

    #reference implementation: Naive Attention calculation
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))# for the causal attention
    P = torch.matmul(Q, K.transpose(2,3)) * softmax_scale
    if causal:
        P[:, :, MASK==0] = float("-inf")
    P = torch.softmax(P.float(), dim=-1).half()# Here the half is for converting float32 to float16.
    ref_O = torch.matmul(P,V)
    ref_O.backward(dO)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None

    # Now, we want to compare the above naive attention calculation with the triton implementation of the Attention
    # Triton Implementation
    tri_out = TritonAttention.apply(Q,K,V, causal, softmax_scale).half()
    tri_out.backward(dO)# We will compute with triton attention
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None

    #Compare
    rtol = 0.0# Relative Distance
    atol = 1e-2# Absolute distance
    # We are comparing that the absolute distance of the corresponding elements of the two tensors is not more than the atol
    assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)

if __name__ == "__main__":
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=2048, HEAD_DIM=64,causal=True)
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=2048, HEAD_DIM=64,causal=False)
    print("PASSED")