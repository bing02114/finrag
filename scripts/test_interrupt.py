
import sqlite3
import sys

from pathlib import Path
from typing import TypedDict

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command


# ==========================================
# 1. Configuration
# ==========================================

DB_PATH = Path(
    "data/checkpoints/hitl_demo.db"
)

DB_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

# THREAD_ID = "finrag-hitl-demo"
THREAD_ID = "finrag-hitl-reject"


# ==========================================
# 2. State
# ==========================================

class ApprovalState(TypedDict):

    request: str

    evidence_ids: list[str]

    human_decision: bool | None

    review_comment: str

    execution_status: str

    final_message: str


# ==========================================
# 3. Human Review Node
# ==========================================

def human_review_node(state: ApprovalState):

    print("\n[Human Review Node]")

    print("Request:", state["request"])

    # Pause execution and wait for human input
    decision = interrupt(
        {
            "type": "approval_request",
            "message": "是否批准继续执行？",
            "request": state["request"],
            "evidence_ids": state["evidence_ids"],
        }
    )

    # This part executes after resume
    print("\n[Human Decision Received]")
    print("Decision:", decision)

    if not isinstance(decision, dict):
        raise ValueError(
            "Resume value must be a dictionary."
        )

    if not isinstance(
        decision.get("approved"),
        bool,
    ):
        raise ValueError(
            "approved must be a boolean."
        )

    return {
        "human_decision": decision["approved"],
        "review_comment": decision.get(
            "comment", ""
        ),
    }


# ==========================================
# 4. Approved Node
# ==========================================

def approved_node(state: ApprovalState):

    print("\n[Approved Node]")

    return {
        "execution_status": "approved",
        "final_message": (
            "人工已经批准，系统可以继续执行。"
        ),
    }


# ==========================================
# 5. Rejected Node
# ==========================================

def rejected_node(state: ApprovalState):

    print("\n[Rejected Node]")

    return {
        "execution_status": "rejected",
        "final_message": (
            "人工拒绝了当前操作，系统停止执行。"
        ),
    }


# ==========================================
# 6. Conditional Routing
# ==========================================

def route_after_review(state: ApprovalState):

    if state["human_decision"]:
        return "approved"

    return "rejected"


# ==========================================
# 7. Build Graph
# ==========================================

def create_graph(checkpointer):

    builder = StateGraph(ApprovalState)

    builder.add_node(
        "human_review",
        human_review_node,
    )

    builder.add_node(
        "approved",
        approved_node,
    )

    builder.add_node(
        "rejected",
        rejected_node,
    )

    builder.add_edge(
        START,
        "human_review",
    )

    builder.add_conditional_edges(
        "human_review",
        route_after_review,
        {
            "approved": "approved",
            "rejected": "rejected",
        },
    )

    builder.add_edge("approved", END)
    builder.add_edge("rejected", END)

    return builder.compile(
        checkpointer=checkpointer
    )


# ==========================================
# 8. Database
# ==========================================

def open_database():

    return sqlite3.connect(
        DB_PATH,
        check_same_thread=False,
        timeout=30,
    )


def get_config():

    return {
        "configurable": {
            "thread_id": THREAD_ID,
        }
    }


# ==========================================
# 9. Pause Mode
# ==========================================

def pause_mode():

    print("\n===== PAUSE MODE =====")

    conn = open_database()

    try:

        checkpointer = SqliteSaver(conn)
        checkpointer.setup()

        graph = create_graph(checkpointer)

        initial_state = {
            "request": (
                "汽轮机调节部套卡涩，"
                "是否批准提交后续处置建议？"
            ),

            "evidence_ids": [
                "D01_第六条:0"
            ],

            "human_decision": None,

            "review_comment": "",

            "execution_status": "pending",

            "final_message": "",
        }

        config = get_config()

        result = graph.invoke(
            initial_state,
            config=config,
        )

        print("\n===== PAUSED STATE =====")

        print(result)

        snapshot = graph.get_state(config)

        print("\nNext:", snapshot.next)

        print(
            "Checkpoint ID:",
            snapshot.config["configurable"]["checkpoint_id"],
        )

        print(
            "\nDatabase:",
            DB_PATH.resolve(),
        )

    finally:

        conn.close()

    print("\nPause mode finished.")


# ==========================================
# 10. Resume Mode
# ==========================================

def resume_mode(approved: bool):

    print("\n===== RESUME MODE =====")

    conn = open_database()

    try:

        checkpointer = SqliteSaver(conn)

        graph = create_graph(checkpointer)

        config = get_config()

        snapshot = graph.get_state(config)

        if not snapshot.values:
            raise RuntimeError(
                "No interrupted state found."
            )

        if not snapshot.next:
            raise RuntimeError(
                "Graph is not waiting for resume."
            )

        print("Current state:")
        print(snapshot.values)

        print("Next:", snapshot.next)

        decision = {
            "approved": approved,
            "comment": (
                "人工确认通过。"
                if approved
                else "人工审核未通过。"
            ),
        }

        result = graph.invoke(
            Command(resume=decision),
            config=config,
        )

        print("\n===== FINAL RESULT =====")

        print("Decision:", result["human_decision"])

        print("Status:", result["execution_status"])

        print("Comment:", result["review_comment"])

        print("Message:", result["final_message"])

        final_snapshot = graph.get_state(config)

        print("\nNext:", final_snapshot.next)

    finally:

        conn.close()

    print("\nResume mode finished.")


# ==========================================
# 11. Main
# ==========================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage: python test_interrupt.py "
            "pause|approve|reject"
        )

        sys.exit(1)

    mode = sys.argv[1]

    if mode == "pause":

        pause_mode()

    elif mode == "approve":

        resume_mode(True)

    elif mode == "reject":

        resume_mode(False)

    else:

        raise ValueError(
            "Mode must be pause, approve or reject."
        )
