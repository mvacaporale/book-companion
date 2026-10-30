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


class StaleConn(FakeConn):
    """Looks open, but the server dropped it while it sat idle."""

    def cursor(self):
        conn = self

        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, sql):
                conn.closed = 2
                raise database.psycopg2.OperationalError("SSL connection has been closed unexpectedly")

        return Cursor()


def test_long_idle_connection_is_pinged_and_replaced(monkeypatch):
    stale, live = StaleConn(), FakeConn()
    pool = FakePool([stale, live])
    monkeypatch.setattr(database, "get_pool", lambda: pool)
    monkeypatch.setitem(database._last_used, id(stale), 0.0)  # idle "forever"

    with database.get_connection() as conn:
        assert conn is live

    assert pool.returned == [(stale, True), (live, False)]
