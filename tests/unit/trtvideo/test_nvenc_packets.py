from trtvideo.video.nvcodec.encoder import write_encoded_packets


def test_buffered_encode_writes_nothing() -> None:
    writes: list[bytes] = []
    write_encoded_packets([], writes.append)
    assert writes == []


def test_encode_and_flush_forward_every_payload_in_bitstream_order() -> None:
    writes: list[bytes] = []
    write_encoded_packets([{"data": b"header", "timestamp": 0, "picture_type": 3}], writes.append)
    write_encoded_packets(
        [
            {"data": b"first", "timestamp": 2, "picture_type": 1},
            {"data": b"second", "timestamp": 1, "picture_type": 2},
        ],
        writes.append,
    )
    assert writes == [b"header", b"first", b"second"]
