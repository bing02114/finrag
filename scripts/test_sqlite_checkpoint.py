import sqlite3
import sys

from pathlib import Path
from typing import TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver


# =====================================
# 1. Configuration
# =====================================

DB_PATH = Path(
    "data/checkpoints/finrag.db"
)

DB_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================
# 2. State
# =====================================

class DemoState(TypedDict):

    query: str

    events: Annotated[
        list[str],
        add,
    ]


# =====================================
# 3. Node
# =====================================

def process_node(state: DemoState):

    query = state["query"]

    print("Processing:", query)

    return {
        "events": [
            f"Processed: {query}"
        ]
    }


# =====================================
# 4. Build Graph
# =====================================

def create_graph(checkpointer):

    builder = StateGraph(DemoState)

    builder.add_node(
        "process",
        process_node,
    )

    builder.add_edge(START, "process")
    builder.add_edge("process", END)

    return builder.compile(
        checkpointer=checkpointer
    )


# =====================================
# 5. SQLite Connection
# =====================================

def open_database():

    conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False,
        timeout=30,
    )

    return conn


# =====================================
# 6. Save Mode
# =====================================

def save_mode():

    print("\n===== SAVE MODE =====")

    conn = open_database()

    try:

        checkpointer = SqliteSaver(conn)

        # Initialize tables if necessary
        checkpointer.setup()

        graph = create_graph(checkpointer)

        config = {
            "configurable": {
                "thread_id": "finrag-sqlite-A"
            }
        }

        result = graph.invoke(
            {
                "query": "变压器乙炔含量标准",
                "events": [],
            },
            config=config,
        )

        print("\nSaved state:")
        print(result)

        snapshot = graph.get_state(config)

        print("\nCheckpoint ID:")
        print(
            snapshot.config["configurable"]["checkpoint_id"]
        )

        print("\nDatabase:")
        print(DB_PATH.resolve())

    finally:

        conn.close()

    print("\nDatabase connection closed.")


# =====================================
# 7. Load Mode
# =====================================

def load_mode():

    print("\n===== LOAD MODE =====")

    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)

    conn = open_database()

    try:

        checkpointer = SqliteSaver(conn)

        graph = create_graph(checkpointer)

        config = {
            "configurable": {
                "thread_id": "finrag-sqlite-A"
            }
        }

        snapshot = graph.get_state(config)

        if not snapshot.values:
            raise RuntimeError(
                "No saved state found."
            )

        print("\nRestored state:")
        print(snapshot.values)

        print("\nNext:")
        print(snapshot.next)

        print("\nCheckpoint history:")

        for item in graph.get_state_history(config):

            print(
                "Checkpoint ID:",
                item.config["configurable"]["checkpoint_id"],
            )

            print(
                "State:",
                item.values,
            )

        assert (
            snapshot.values["query"]
            == "变压器乙炔含量标准"
        )

        assert len(
            snapshot.values["events"]
        ) == 1

    finally:

        conn.close()

    print("\nSQLite persistence test passed!")


# =====================================
# 8. Main
# =====================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage: python test_sqlite_checkpoint.py save|load"
        )

        sys.exit(1)

    mode = sys.argv[1]

    if mode == "save":

        save_mode()

    elif mode == "load":

        load_mode()

    else:

        raise ValueError(
            "Mode must be save or load."
        )
