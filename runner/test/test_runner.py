from precise_runner import ReadWriteStream


class TestReadWriteStream:
    def test_read_write(self):
        s = ReadWriteStream(b'1234567890')
        assert s.read(2) == b'12'
        assert s.read(2) == b'34'
        s.write(b'hi')
        assert s.read() == b'567890hi'
        s.write(b'hello')
        assert s.read() == b'hello'
        assert s.read(1, timeout=0.1) == b''

    def test_chop(self):
        s = ReadWriteStream(chop_samples=10)
        s.write(b'1234567890hello')
        assert s.read(5) == b'hello'
