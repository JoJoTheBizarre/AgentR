from langgraph.checkpoint.postgres import PostgresSaver


class StateCheckpointer:
    def __init__(self, db_uri: str, thread_id: str):
        self.db_uri = db_uri
        self._saver = None
        self.thread_id = thread_id

    def __enter__(self):
        self._saver = PostgresSaver.from_conn_string(self.db_uri)
        self._saver.__enter__()  # enter the context
        self._saver.setup()  # run setup once
        return self._saver

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._saver:
            self._saver.__exit__(exc_type, exc_val, exc_tb)
