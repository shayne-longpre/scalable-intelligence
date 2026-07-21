from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


WIDTH = 1200
INK = "#182026"
MUTED = "#5E6A71"
GRID = "#DCE2E5"
SOL = "#2364AA"
FABLE = "#C44536"
IDEAL = "#88949A"
ACCENT = "#8F5D2E"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create publication figures from a catalog-ladder report card."
    )
    parser.add_argument("--report-summary", required=True)
    parser.add_argument("--catalog", default="data/model_catalog.openrouter.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--inline-html")
    args = parser.parse_args()

    report = _load_json(Path(args.report_summary))
    catalog = _load_json(Path(args.catalog))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog_names = {
        item["provider_model_id"]: item.get("display_name") or item["provider_model_id"]
        for item in catalog["models"]
    }
    runs = [_run_data(run, catalog_names) for run in report["runs"]]
    if len(runs) == 2:
        outputs = {
            "predicted_vs_external": output_dir / "predicted-vs-external.svg",
            "discrimination_by_gap": output_dir / "discrimination-by-gap.svg",
            "evidence_scaling": output_dir / "evidence-scaling.svg",
        }
        _write_svg(outputs["predicted_vs_external"], _predicted_vs_external(runs))
        _write_svg(outputs["discrimination_by_gap"], _discrimination_by_gap(runs))
        _write_svg(outputs["evidence_scaling"], _evidence_scaling(runs))
        if args.inline_html:
            _write_inline_html(Path(args.inline_html), outputs)
    elif len(runs) == 4:
        if args.inline_html:
            raise ValueError("inline HTML is only available for the two-run figure set")
        outputs = {"crossed_judges": output_dir / "crossed-judge-accuracy.svg"}
        _write_svg(outputs["crossed_judges"], _crossed_judge_accuracy(runs))
    else:
        raise ValueError("catalog ladder figures require two primary or four crossed runs")
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))
    return 0


def _run_data(run: dict, catalog_names: dict[str, str]) -> dict:
    participants = {
        item["id"]: {
            "provider_model_id": item["provider_model_id"],
            "name": _short_name(catalog_names.get(item["provider_model_id"], item["id"])),
        }
        for item in run["participants"]
    }
    final = run["probe_budget_results"][-1]
    rank_by_id = {participant_id: index for index, participant_id in enumerate(final["ranking"], 1)}
    scores = run["prior_participant_scores"]
    reported = set(run.get("prior_reported_score_participants", []))
    ordered_ids = sorted(scores, key=lambda participant_id: (-scores[participant_id], participant_id))
    true_rank = {participant_id: index for index, participant_id in enumerate(ordered_ids, 1)}
    points = []
    for participant_id in ordered_ids:
        points.append(
            {
                "id": participant_id,
                "name": participants[participant_id]["name"],
                "score": float(scores[participant_id]),
                "predicted_rank": rank_by_id[participant_id],
                "true_rank": true_rank[participant_id],
                "reported": participant_id in reported,
            }
        )
    judge_model = run["judges"][0]["provider_model_id"]
    return {
        "name": run["name"],
        "judge": "Sol" if "sol" in judge_model.lower() else "Fable",
        "evidence": _evidence_author(run["name"], judge_model),
        "points": points,
        "checkpoints": run["probe_budget_results"],
    }


def _evidence_author(run_name: str, judge_model: str) -> str:
    normalized = run_name.lower()
    if "sol_evidence" in normalized:
        return "Sol"
    if "fable_evidence" in normalized:
        return "Fable"
    return "Sol" if "sol" in judge_model.lower() else "Fable"


def _predicted_vs_external(runs: list[dict]) -> ET.Element:
    height = 610
    root = _svg_root(height)
    _heading(
        root,
        "Can an AI judge recover the capability ladder?",
        "Each point is one anonymous model. Higher judged percentile means the judge ranked it as more capable.",
    )
    panel_width = 500
    panel_height = 390
    panel_y = 130
    panel_xs = [80, 660]
    for run, panel_x in zip(runs, panel_xs, strict=True):
        color = SOL if run["judge"] == "Sol" else FABLE
        _text(root, panel_x, 105, run["judge"], css_class="panel-title")
        _text(
            root,
            panel_x + panel_width,
            105,
            f"pairwise {run['checkpoints'][-1]['pairwise_accuracy']:.1%}",
            css_class="panel-metric",
            anchor="end",
        )
        _axes(
            root,
            panel_x,
            panel_y,
            panel_width,
            panel_height,
            x_ticks=[0, 10, 20, 30, 40, 50, 60],
            y_ticks=[0, 25, 50, 75, 100],
            x_domain=(0, 62),
            y_domain=(0, 100),
            x_label="",
            y_label="Judged capability percentile" if panel_x == panel_xs[0] else None,
            y_percent=True,
        )
        count = len(run["points"])
        perfect = sorted(run["points"], key=lambda item: item["score"])
        path = []
        for point in perfect:
            true_percentile = 100 * (count - point["true_rank"]) / (count - 1)
            x = _scale(point["score"], 0, 62, panel_x, panel_x + panel_width)
            y = _scale(true_percentile, 0, 100, panel_y + panel_height, panel_y)
            path.append((x, y))
        _polyline(root, path, stroke=IDEAL, css_class="ideal-line")

        biggest_miss = max(
            run["points"],
            key=lambda item: abs(item["predicted_rank"] - item["true_rank"]),
        )
        top_names = {item["name"] for item in run["points"][:2]}
        label_names = top_names | {biggest_miss["name"]}
        for point in run["points"]:
            percentile = 100 * (count - point["predicted_rank"]) / (count - 1)
            x = _scale(point["score"], 0, 62, panel_x, panel_x + panel_width)
            y = _scale(percentile, 0, 100, panel_y + panel_height, panel_y)
            circle = ET.SubElement(
                root,
                "circle",
                {
                    "cx": f"{x:.1f}",
                    "cy": f"{y:.1f}",
                    "r": "4.5" if point["reported"] else "5.5",
                    "fill": color if point["reported"] else "none",
                    "stroke": color,
                    "stroke-width": "1.6",
                    "opacity": "0.84",
                },
            )
            ET.SubElement(circle, "title").text = (
                f"{point['name']}: external {point['score']:.2f}; "
                f"judge rank {point['predicted_rank']} of {count}"
            )
            if point["name"] in label_names:
                dx, dy, anchor = _label_offset(point["name"], biggest_miss["name"])
                if point["name"] == biggest_miss["name"]:
                    ET.SubElement(
                        root,
                        "circle",
                        {
                            "cx": f"{x:.1f}",
                            "cy": f"{y:.1f}",
                            "r": "8",
                            "fill": "none",
                            "stroke": ACCENT,
                            "stroke-width": "1.5",
                        },
                    )
                _text(
                    root,
                    x + dx,
                    y + dy,
                    point["name"],
                    css_class="point-label",
                    anchor=anchor,
                )
    _text(root, 600, 548, "External Intelligence Index", css_class="axis-label", anchor="middle")
    _line(root, 315, 580, 345, 580, stroke=IDEAL, css_class="ideal-line")
    _text(root, 353, 584, "perfect ordering", css_class="legend")
    _circle(root, 485, 580, 4.5, fill=INK, stroke=INK)
    _text(root, 497, 584, "reported score", css_class="legend")
    _circle(root, 625, 580, 5.5, fill="none", stroke=INK)
    _text(root, 637, 584, "estimated weak anchor", css_class="legend")
    return root


def _discrimination_by_gap(runs: list[dict]) -> ET.Element:
    height = 570
    root = _svg_root(height)
    _heading(
        root,
        "Large capability differences are easy; close calls are not",
        "Final pairwise accuracy after six probes, using the 47 models with reported external scores.",
    )
    x0, y0, width, height_plot = 120, 120, 920, 360
    labels = [item["label"] for item in runs[0]["checkpoints"][-1]["pairwise_accuracy_by_score_gap"]]
    counts = [item["pair_count"] for item in runs[0]["checkpoints"][-1]["pairwise_accuracy_by_score_gap"]]
    xs = [x0 + index * width / (len(labels) - 1) for index in range(len(labels))]
    _axes(
        root,
        x0,
        y0,
        width,
        height_plot,
        x_ticks=[],
        y_ticks=[50, 60, 70, 80, 90, 100],
        x_domain=(0, 1),
        y_domain=(45, 100),
        x_label="Difference in external Intelligence Index",
        y_label="Correctly ordered pairs",
        y_percent=True,
    )
    chance_y = _scale(50, 45, 100, y0 + height_plot, y0)
    _line(root, x0, chance_y, x0 + width, chance_y, stroke=IDEAL, css_class="chance-line")
    _text(root, x0 + width, chance_y - 8, "chance", css_class="legend", anchor="end")
    for x, label, count in zip(xs, labels, counts, strict=True):
        _text(root, x, y0 + height_plot + 30, label, css_class="tick", anchor="middle")
        _text(root, x, y0 + height_plot + 49, f"n={count}", css_class="tick-muted", anchor="middle")
    for run in runs:
        color = SOL if run["judge"] == "Sol" else FABLE
        values = [
            item["accuracy"] * 100
            for item in run["checkpoints"][-1]["pairwise_accuracy_by_score_gap"]
        ]
        points = [
            (x, _scale(value, 45, 100, y0 + height_plot, y0))
            for x, value in zip(xs, values, strict=True)
        ]
        _polyline(root, points, stroke=color, css_class="series-line")
        for x, y, value in zip(xs, [point[1] for point in points], values, strict=True):
            if run["judge"] == "Sol":
                _circle(root, x, y, 6, fill=color, stroke=color)
            else:
                _square(root, x, y, 11, fill=color, stroke=color)
            _text(
                root,
                x,
                y - 13 if run["judge"] == "Sol" else y + 24,
                f"{value:.0f}%",
                css_class="value-label",
                anchor="middle",
            )
        end_y = points[-1][1]
        _text(
            root,
            xs[-1] + 18,
            end_y - 8 if run["judge"] == "Sol" else end_y + 20,
            run["judge"],
            css_class="series-label",
        )
    return root


def _evidence_scaling(runs: list[dict]) -> ET.Element:
    height = 560
    root = _svg_root(height)
    _heading(
        root,
        "More probing helped one judge, but not the other",
        "Overall pairwise accuracy after the four-probe opening and each targeted adaptive follow-up.",
    )
    x0, y0, width, height_plot = 140, 120, 860, 340
    probe_counts = [item["probe_count"] for item in runs[0]["checkpoints"]]
    xs = [x0 + index * width / (len(probe_counts) - 1) for index in range(len(probe_counts))]
    _axes(
        root,
        x0,
        y0,
        width,
        height_plot,
        x_ticks=[],
        y_ticks=[78, 80, 82, 84, 86],
        x_domain=(0, 1),
        y_domain=(77, 86),
        x_label="Cumulative probes",
        y_label="Correctly ordered pairs",
        y_percent=True,
    )
    for x, count in zip(xs, probe_counts, strict=True):
        _text(root, x, y0 + height_plot + 32, str(count), css_class="tick", anchor="middle")
        phase = "opening" if count == probe_counts[0] else f"adaptive {count - probe_counts[0]}"
        _text(root, x, y0 + height_plot + 52, phase, css_class="tick-muted", anchor="middle")
    for run in runs:
        color = SOL if run["judge"] == "Sol" else FABLE
        values = [item["pairwise_accuracy"] * 100 for item in run["checkpoints"]]
        points = [
            (x, _scale(value, 77, 86, y0 + height_plot, y0))
            for x, value in zip(xs, values, strict=True)
        ]
        _polyline(root, points, stroke=color, css_class="series-line")
        for x, y, value in zip(xs, [point[1] for point in points], values, strict=True):
            if run["judge"] == "Sol":
                _circle(root, x, y, 6, fill=color, stroke=color)
            else:
                _square(root, x, y, 11, fill=color, stroke=color)
            _text(root, x, y - 14, f"{value:.1f}%", css_class="value-label", anchor="middle")
        delta = values[-1] - values[0]
        _text(
            root,
            points[-1][0] + 22,
            points[-1][1] + (20 if run["judge"] == "Sol" else 8),
            f"{run['judge']}  {delta:+.1f} points",
            css_class="series-label",
        )
    return root


def _crossed_judge_accuracy(runs: list[dict]) -> ET.Element:
    height = 590
    root = _svg_root(height)
    _heading(
        root,
        "The questions mattered as much as the judge",
        "Two judges, identical frozen answers. Accuracy uses 47 models with reported external scores.",
    )
    panel_width = 440
    panel_height = 330
    panel_y = 135
    panel_xs = [105, 655]
    for evidence, panel_x in zip(("Sol", "Fable"), panel_xs, strict=True):
        panel_runs = [run for run in runs if run["evidence"] == evidence]
        if {run["judge"] for run in panel_runs} != {"Sol", "Fable"}:
            raise ValueError(f"crossed figure is missing one judge for {evidence} evidence")
        _text(root, panel_x, 108, f"{evidence}-authored evidence", css_class="panel-title")
        probe_counts = [item["probe_count"] for item in panel_runs[0]["checkpoints"]]
        xs = [
            panel_x + index * panel_width / (len(probe_counts) - 1)
            for index in range(len(probe_counts))
        ]
        _axes(
            root,
            panel_x,
            panel_y,
            panel_width,
            panel_height,
            x_ticks=[],
            y_ticks=[78, 80, 82, 84, 86, 88],
            x_domain=(0, 1),
            y_domain=(78, 88.5),
            x_label="",
            y_label="Correctly ordered pairs" if evidence == "Sol" else None,
            y_percent=True,
        )
        for x, count in zip(xs, probe_counts, strict=True):
            _text(root, x, panel_y + panel_height + 28, str(count), css_class="tick", anchor="middle")
        for run in sorted(panel_runs, key=lambda item: item["judge"], reverse=True):
            color = SOL if run["judge"] == "Sol" else FABLE
            values = [item["pairwise_accuracy"] * 100 for item in run["checkpoints"]]
            points = [
                (x, _scale(value, 78, 88.5, panel_y + panel_height, panel_y))
                for x, value in zip(xs, values, strict=True)
            ]
            _polyline(root, points, stroke=color, css_class="series-line")
            for index, (x, y, value) in enumerate(
                zip(xs, [point[1] for point in points], values, strict=True)
            ):
                if run["judge"] == "Sol":
                    _circle(root, x, y, 6, fill=color, stroke=color)
                    label_y = y + 23 if index == len(points) - 1 else y - 13
                else:
                    _square(root, x, y, 11, fill=color, stroke=color)
                    label_y = y - 13 if index == len(points) - 1 else y + 23
                _text(root, x, label_y, f"{value:.1f}%", css_class="value-label", anchor="middle")
            _text(
                root,
                points[-1][0] + 15,
                points[-1][1] + (20 if run["judge"] == "Sol" else -8),
                run["judge"],
                css_class="series-label",
            )
    _text(root, 600, 525, "Cumulative probes", css_class="axis-label", anchor="middle")
    _circle(root, 445, 563, 6, fill=SOL, stroke=SOL)
    _text(root, 458, 567, "Sol judge", css_class="legend")
    _square(root, 575, 563, 11, fill=FABLE, stroke=FABLE)
    _text(root, 589, 567, "Fable judge", css_class="legend")
    return root


def _svg_root(height: int) -> ET.Element:
    root = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": str(WIDTH),
            "height": str(height),
            "viewBox": f"0 0 {WIDTH} {height}",
            "role": "img",
        },
    )
    ET.SubElement(root, "style").text = """
        text { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #182026; letter-spacing: 0; }
        .title { font-size: 27px; font-weight: 500; }
        .subtitle { font-size: 15px; fill: #5E6A71; }
        .panel-title { font-size: 19px; font-weight: 500; }
        .panel-metric, .series-label { font-size: 14px; font-weight: 500; }
        .tick, .point-label, .value-label { font-size: 12px; }
        .tick-muted, .legend { font-size: 12px; fill: #5E6A71; }
        .axis-label { font-size: 14px; fill: #38454B; }
        .grid { stroke: #DCE2E5; stroke-width: 1; }
        .axis { stroke: #879399; stroke-width: 1.2; }
        .ideal-line { fill: none; stroke-dasharray: 5 5; stroke-width: 1.5; }
        .chance-line { fill: none; stroke-dasharray: 4 5; stroke-width: 1.2; }
        .series-line { fill: none; stroke-width: 2.5; stroke-linejoin: round; stroke-linecap: round; }
    """
    return root


def _heading(root: ET.Element, title: str, subtitle: str) -> None:
    ET.SubElement(root, "title").text = title
    ET.SubElement(root, "desc").text = subtitle
    _text(root, 60, 43, title, css_class="title")
    _text(root, 60, 71, subtitle, css_class="subtitle")


def _axes(
    root: ET.Element,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    x_ticks: list[float],
    y_ticks: list[float],
    x_domain: tuple[float, float],
    y_domain: tuple[float, float],
    x_label: str,
    y_label: str | None,
    y_percent: bool,
) -> None:
    for tick in x_ticks:
        px = _scale(tick, *x_domain, x, x + width)
        _line(root, px, y, px, y + height, css_class="grid")
        _text(root, px, y + height + 23, str(tick), css_class="tick", anchor="middle")
    for tick in y_ticks:
        py = _scale(tick, *y_domain, y + height, y)
        _line(root, x, py, x + width, py, css_class="grid")
        label = f"{tick:.0f}%" if y_percent else f"{tick:g}"
        _text(root, x - 12, py + 4, label, css_class="tick", anchor="end")
    _line(root, x, y + height, x + width, y + height, css_class="axis")
    _line(root, x, y, x, y + height, css_class="axis")
    if x_label:
        _text(
            root,
            x + width / 2,
            y + height + 70,
            x_label,
            css_class="axis-label",
            anchor="middle",
        )
    if y_label:
        label = _text(root, x - 66, y + height / 2, y_label, css_class="axis-label", anchor="middle")
        label.set("transform", f"rotate(-90 {x - 66:.1f} {y + height / 2:.1f})")


def _text(
    root: ET.Element,
    x: float,
    y: float,
    value: str,
    *,
    css_class: str,
    anchor: str = "start",
) -> ET.Element:
    element = ET.SubElement(
        root,
        "text",
        {
            "x": f"{x:.1f}",
            "y": f"{y:.1f}",
            "class": css_class,
            "text-anchor": anchor,
        },
    )
    element.text = value
    return element


def _line(
    root: ET.Element,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str | None = None,
    css_class: str,
) -> None:
    attrs = {
        "x1": f"{x1:.1f}",
        "y1": f"{y1:.1f}",
        "x2": f"{x2:.1f}",
        "y2": f"{y2:.1f}",
        "class": css_class,
    }
    if stroke:
        attrs["stroke"] = stroke
    ET.SubElement(root, "line", attrs)


def _polyline(
    root: ET.Element,
    points: list[tuple[float, float]],
    *,
    stroke: str,
    css_class: str,
) -> None:
    ET.SubElement(
        root,
        "polyline",
        {
            "points": " ".join(f"{x:.1f},{y:.1f}" for x, y in points),
            "stroke": stroke,
            "class": css_class,
        },
    )


def _circle(
    root: ET.Element,
    x: float,
    y: float,
    radius: float,
    *,
    fill: str,
    stroke: str,
) -> None:
    ET.SubElement(
        root,
        "circle",
        {
            "cx": f"{x:.1f}",
            "cy": f"{y:.1f}",
            "r": f"{radius:.1f}",
            "fill": fill,
            "stroke": stroke,
            "stroke-width": "1.5",
        },
    )


def _square(
    root: ET.Element,
    x: float,
    y: float,
    size: float,
    *,
    fill: str,
    stroke: str,
) -> None:
    ET.SubElement(
        root,
        "rect",
        {
            "x": f"{x - size / 2:.1f}",
            "y": f"{y - size / 2:.1f}",
            "width": f"{size:.1f}",
            "height": f"{size:.1f}",
            "fill": fill,
            "stroke": stroke,
            "stroke-width": "1.5",
        },
    )


def _label_offset(name: str, biggest_miss: str) -> tuple[int, int, str]:
    if name == biggest_miss:
        return (10, -10, "start")
    offsets = {
        "Claude Fable 5": (-8, -12, "end"),
        "GPT-5.6 Sol": (-8, 17, "end"),
        "Kimi K3": (-8, 31, "end"),
    }
    return offsets.get(name, (9, -9, "start"))


def _short_name(name: str) -> str:
    replacements = {
        "Anthropic: ": "",
        "OpenAI: ": "",
        "MoonshotAI: ": "",
        "Google: ": "",
        "Meta: ": "",
    }
    for prefix, replacement in replacements.items():
        name = name.replace(prefix, replacement)
    return name


def _scale(
    value: float,
    domain_min: float,
    domain_max: float,
    range_min: float,
    range_max: float,
) -> float:
    return range_min + (value - domain_min) * (range_max - range_min) / (
        domain_max - domain_min
    )


def _write_svg(path: Path, root: ET.Element) -> None:
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _write_inline_html(path: Path, outputs: dict[str, Path]) -> None:
    plots = []
    for name in ("predicted_vs_external", "discrimination_by_gap", "evidence_scaling"):
        svg = outputs[name].read_text(encoding="utf-8")
        svg = svg[svg.index("<svg") :]
        for source, target in (
            (INK, "var(--foreground)"),
            (MUTED, "var(--muted-foreground)"),
            (GRID, "var(--border)"),
            (SOL, "var(--viz-series-1)"),
            (FABLE, "var(--viz-series-2)"),
            (IDEAL, "var(--muted-foreground)"),
            (ACCENT, "var(--viz-series-3)"),
            ("#38454B", "var(--foreground)"),
            ("#879399", "var(--muted-foreground)"),
        ):
            svg = svg.replace(source, target)
        plots.append(svg)
    fragment = f"""<div id="catalog-ladder-results" class="catalog-ladder-results">
  <section class="catalog-ladder-primary">{plots[0]}</section>
  <div class="catalog-ladder-secondary">
    <section>{plots[1]}</section>
    <section>{plots[2]}</section>
  </div>
</div>
<style>
  #catalog-ladder-results {{ width: 100%; color: var(--foreground); }}
  #catalog-ladder-results section {{ min-width: 0; }}
  #catalog-ladder-results svg {{ display: block; width: 100%; height: auto; }}
  #catalog-ladder-results .catalog-ladder-secondary {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 24px;
    margin-top: 20px;
  }}
</style>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fragment, encoding="utf-8")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    raise SystemExit(main())
