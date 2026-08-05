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