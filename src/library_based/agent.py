import json

import torch
from transformers import pipeline

from .utils import sanitize_for_json


def generate_dispatch_with_hf_agent(optimised_journeys: list) -> str:
    """
    Generates a SmartFlow daily rebalancing report using an agentic AI approach.

    This function performs the following steps:
      1\. Generates a natural language Manager's Briefing using a Hugging Face LLM.
      2\. Builds deterministic, reliable Markdown tables directly from the source data.
      3\. Falls back to a deterministic Markdown report if the LLM step fails.

    Args:
        optimised_journeys (list): List of journey dictionaries, each containing
            'truck_id' and a list of 'legs'. Each leg should have
            'dispatch_time', 'src', 'target', and 'move' keys.

    Returns:
        str: Markdown-formatted report including the Manager's Briefing and dispatch tickets for each truck.
    """

    def fallback_formatter(journeys):
        """
        Generates a fallback Markdown report for SmartFlow daily rebalancing.

        Args:
            journeys (list): List of journey dictionaries, each containing
                'truck_id' and a list of 'legs'. Each leg should have
                'dispatch_time', 'src', 'target', and 'move' keys.

        Returns:
            str: Markdown-formatted report summarizing dispatch tickets for each truck.
        """
        report = ["## SmartFlow Daily Rebalancing Report (Fallback)", ""]
        for journey in journeys:
            report.append(f"### 📋 Dispatch Ticket: Truck {journey['truck_id']}")
            report.append("| Leg | Dispatch Time | From | To | Action |")
            report.append("|----:|---------------|------|----|--------|")
            for i, leg in enumerate(journey["legs"], start=1):
                dt = leg.get("dispatch_time", "N/A")
                src = leg.get("src", "N/A")
                target = leg.get("target", "N/A")
                mv = leg.get("move", 0)
                action = f"Move {mv} bike{'s' if mv != 1 else ''}"
                report.append(f"| {i} | {dt} | {src} | {target} | {action} |")
            report.append("")
        return "\n".join(report)

    # Sanitize first to have clean data for all subsequent steps.
    clean_journeys = sanitize_for_json(optimised_journeys)

    try:
        # 1) Load the LLM
        pipe = pipeline(
            "text-generation",
            model="google/gemma-2b-it",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # 2) Serialize data for the briefing prompt
        raw_json = json.dumps(clean_journeys, indent=2)

        # 3) Manager’s Briefing prompt (This is the ONLY task for the LLM now)
        briefing_prompt = (
            "<start_of_turn>user\n"
            "You are SmartFlow, an autonomous Logistics Analyst AI.\n"
            "Based on the following dispatch data, provide an 8-point summary covering performance, outlook, truck count, efficiencies, and potential risks.\n"
            "**REAL INPUT DATA:**\n"
            "```json\n"
            f"{raw_json}\n"
            "```\n"
            "Output exactly 8 bullet points, each starting with a hyphen. Each bullet point should have a title and a description.\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        brief_out = pipe(briefing_prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]
        briefing = brief_out.split("<start_of_turn>model", 1)[1].strip()

        # --- THE SECOND LLM CALL HAS BEEN ENTIRELY REMOVED ---

        # 4) Assemble final markdown using the ORIGINAL, RELIABLE data
        md = [
            "## SmartFlow Daily Rebalancing Report",
            "",
            "### 🧑‍💼 Manager's Briefing",
            briefing,
            "",
            "---",
            ""
        ]

        # Iterate over the clean_journeys data directly
        for truck in clean_journeys:
            tid = truck["truck_id"]
            md.append(f"### 📋 Dispatch Ticket: Truck {tid}")
            md.append("| Leg | Dispatch Time | From | To | Action |")
            md.append("|----:|---------------|------|----|--------|")

            # Use enumerate to generate the leg number reliably
            for i, leg in enumerate(truck["legs"], start=1):
                dt = leg.get("dispatch_time", "N/A")
                fr = leg.get("src", "N/A")  # Use original keys
                to = leg.get("tgt", "N/A")  # Use original keys
                mv = leg.get("move", 0)
                action = f"Move {mv} bike{'s' if mv != 1 else ''}"

                md.append(f"| {i} | {dt} | {fr} | {to} | {action} |")
            md.append("")
        return "\n".join(md)

    except Exception as e:
        # The exception is now almost certainly from the briefing step
        print(f"⚠️ Agentic AI briefing failed ({e}), using fallback for full report.")
        return fallback_formatter(clean_journeys)
