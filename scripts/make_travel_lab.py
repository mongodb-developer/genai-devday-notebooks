"""Generate labs/travel-agent-lab.ipynb from solutions/travel-agent-lab.ipynb.

The lab version replaces selected lines with auto-numbered <CODE_BLOCK_n>
placeholders. Deriving the lab from the solution guarantees the two never drift
and that blank numbering is contiguous.

Each entry in BLANKS is (cell_index, exact_snippet_from_solution). The snippet is
replaced in place, so surrounding comments and hints survive untouched.

Usage:
    python scripts/make_travel_lab.py
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "solutions" / "travel-agent-lab.ipynb"
LAB = ROOT / "labs" / "travel-agent-lab.ipynb"

# Ordered list of (cell_index, snippet_to_blank).
# Order determines the CODE_BLOCK numbering, so keep it top-to-bottom.
BLANKS: list[tuple[int, str]] = [
    # --- Step 3: vector index with filter fields ---
    (12, '            {"type": "filter", "path": "address.market"},\n'
         '            {"type": "filter", "path": "accommodates"},'),
    # --- Step 4a: the graphLookup traversal ---
    (18, '            "startWith": "$dst_airport",\n'
         '            "connectFromField": "dst_airport",\n'
         '            "connectToField": "src_airport",'),
    (18, '            "restrictSearchWithMatch": {"dst_airport": "OPO"},'),
    # --- Step 4a: the index that makes it fast ---
    (20, 'create_index(routes_collection, [("src_airport", 1)], "src_airport_1")'),
    # --- Step 5: query embedding ---
    (27, '    return vo.embed([query], model=model, input_type="query").embeddings[0]'),
    # --- Tool 1: the one-stop graph traversal ---
    (31, '                        "startWith": "$dst_airport",\n'
         '                        "connectFromField": "dst_airport",\n'
         '                        "connectToField": "src_airport",'),
    (31, '                        "restrictSearchWithMatch": {\n'
         '                            "dst_airport": {"$in": destinations}\n'
         '                        },'),
    # --- Tool 2: filtered vector search ---
    (33, '    vector_filter = {\n'
         '        "address.market": destination_city.title(),\n'
         '        "accommodates": {"$gte": party},\n'
         '    }'),
    (33, '                "filter": vector_filter,'),
    (33, '        {"$match": {"availability.availability_365": {"$gt": 0}}},'),
    (33, '        pipeline.append({"$match": {"amenities": {"$in": CHILD_AMENITIES}}})'),
    # --- Tool 4: the date-overlap dedupe query ---
    (37, '                "booking_dates.check_in": {"$lt": check_out},\n'
         '                "booking_dates.check_out": {"$gt": check_in},'),
    # --- Tool 5: validation + idempotent write ---
    (40, '    min_nights = int(stay.get("minimum_nights") or 1)'),
    (40, '    drafts_collection.replace_one({"_id": draft["_id"]}, draft, upsert=True)'),
    # --- Step 6: bind tools ---
    (43, "llm_with_tools = llm.bind_tools(tools)"),
    # --- Step 7: typed state ---
    (47, "    messages: Annotated[list, add_messages]"),
    (47, "    trip: TripRequest"),
    (47, "    approved: bool"),
    # --- Step 8: tool node ---
    (50, '        observation = selected_tool.invoke(tool_call["args"])'),
    # --- Step 8: the interrupt itself ---
    (52, '    answer = interrupt(\n'
         '        {\n'
         '            "question": "Create this booking draft?",\n'
         '            "booking": args,\n'
         '        }\n'
         '    )'),
    (52, '    observation = tools_by_name["create_booking_draft"].invoke(args)'),
    # --- Step 9: routing that makes the gate structural ---
    (54, '    if any(call["name"] == "create_booking_draft" for call in tool_calls):\n'
         '        return "confirm"\n'
         '    return "tools"'),
    (55, 'graph.add_node("confirm", confirm)'),
    (55, 'graph.add_conditional_edges(\n'
         '    "agent",\n'
         '    route_tools,\n'
         '    {"tools": "tools", "confirm": "confirm", END: END},\n'
         ')'),
    # --- Step 9: checkpointer ---
    (57, "app = graph.compile(checkpointer=checkpointer)"),
    # --- Step 10: resuming an interrupt ---
    (59, '    for step in app.stream(Command(resume=answer), config, stream_mode="values"):'),
    # --- Step 11: long-term memory ---
    (81, '    hits = mongodb_store.search((USER_ID,), query=str(messages[-1].content), limit=10)'),
    (82, "app = graph.compile(checkpointer=checkpointer, store=mongodb_store)"),
]


def main() -> int:
    notebook = json.loads(SOLUTION.read_text())
    cells = notebook["cells"]

    for number, (index, snippet) in enumerate(BLANKS, start=1):
        cell = cells[index]
        if cell["cell_type"] != "code":
            raise SystemExit(f"cell {index} is not code")
        source = "".join(cell["source"])
        if snippet not in source:
            raise SystemExit(
                f"CODE_BLOCK_{number}: snippet not found in cell {index}:\n{snippet[:120]}"
            )
        if source.count(snippet) > 1:
            raise SystemExit(f"CODE_BLOCK_{number}: snippet is ambiguous in cell {index}")

        # Preserve the indentation of the first replaced line
        indent = snippet[: len(snippet) - len(snippet.lstrip())]
        source = source.replace(snippet, f"{indent}<CODE_BLOCK_{number}>", 1)
        lines = source.split("\n")
        cell["source"] = [f"{line}\n" for line in lines[:-1]] + [lines[-1]]

    LAB.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"wrote {LAB.relative_to(ROOT)} with {len(BLANKS)} blanks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
