import torch
import torch.distributed as dist


def _ring_step(send, recv):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    request = dist.isend(send, (rank + 1) % world_size)
    dist.recv(recv, (rank - 1) % world_size)
    request.wait()


def ring_reduce_scatter(tensor):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    chunks = [chunk.clone() for chunk in tensor.chunk(world_size)]

    buffer = torch.empty_like(chunks[0])
    for step in range(world_size - 1):
        send_index = (rank - step - 1) % world_size
        recv_index = (rank - step - 2) % world_size
        _ring_step(chunks[send_index], buffer)
        chunks[recv_index] += buffer
    return chunks[rank]


def ring_all_gather(shard):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    chunks = [torch.empty_like(shard) for _ in range(world_size)]
    chunks[rank].copy_(shard)

    for step in range(world_size - 1):
        send_index = (rank - step) % world_size
        recv_index = (rank - step - 1) % world_size
        _ring_step(chunks[send_index], chunks[recv_index])
    return torch.cat(chunks)


def ring_all_reduce(tensor):
    return ring_all_gather(ring_reduce_scatter(tensor))
