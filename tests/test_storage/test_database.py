"""get_connection replaces connections the server already closed."""

from book_companion.storage import database


class FakeConn:
    def __init__(self, closed=0):
        self.closed = closed
        self.committed = False

    def commit(self):
        self.committed = True

    def rollback(self):
        if self.closed:
            raise RuntimeError("connection already closed")


class FakePool:
    def __init__(self, conns):
        self.conns = list(conns)
        self.returned = []

    def getconn(self):
        return self.conns.pop(0)

    def putconn(self, conn, close=False):
        self.returned.append((conn, close))


def test_closed_connection_is_discarded(monkeypatch):
    dead, live = FakeConn(closed=2), FakeConn()
    pool = FakePool([dead, live])
    monkeypatch.setattr(database, "get_pool", lambda: pool)

    with database.get_connection() as conn:
        assert conn is live

    assert live.committed
    assert pool.returned == [(dead, True), (live, False)]
