import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph import SupportAgentGraph

def main():

    graph = SupportAgentGraph()

    while True:
        try:
            q = input("Enter customer question (type 'exit' to quit): ").strip()
            if not q:
                continue
            if q.lower() in ["exit", "quit", "q"]:
                break

            result = graph.run(q)

            trace = result.pop("node_trace", [])
            
            print("\n" + "-" * 40)
            print("GRAPH EXECUTION TRACE:")
            print("-" * 40)
            print(" -> ".join(trace))

            print("\n" + "-" * 40)
            print("READABLE RESPONSE:")
            print("-" * 40)
            print(result["answer"])

            print("\n" + "-" * 40)
            print("STRUCTURED JSON OUTPUT:")
            print("-" * 40)
            print(json.dumps(result, indent=2))
            print("=" * 60 + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()


    """I am a read-only Viewer. Can I create an API credential for a reporting script?"""
    """Our data sync is not working. Can you tell me how to fix it?"""
    """I am unhappy and want my money returned to my card."""