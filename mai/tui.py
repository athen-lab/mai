"""Curses workbench for the Hugging Face-backed MAI dataset pipeline."""

from __future__ import annotations

from collections import deque
import curses
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import textwrap
import threading
from typing import Any, Callable

from .dataset import PipelineError, list_spec_groups
from .forms import (
    BuildForm,
    InitForm,
    PublishForm,
    PullForm,
    ValidateForm,
    command_preview,
    resolve,
)
from .hub import list_remote_groups


Field = tuple[str, str, str, str]
EXTENDED_DARK_FOREGROUND = 235

OPERATIONS: list[tuple[str, str, str]] = [
    (
        "build",
        "Build dataset",
        "Acquire, generate, and build a validated local Hugging Face package.",
    ),
    (
        "init",
        "Initialize build spec",
        "Create an empty provenance-first dataset specification.",
    ),
    (
        "validate",
        "Validate package",
        "Audit metadata, group balance, checksums, and normalized images.",
    ),
    (
        "publish",
        "Publish to Hugging Face",
        "Upload a validated package and optionally create a release tag.",
    ),
    (
        "pull",
        "Download from Hugging Face",
        "Retrieve exact semantic groups from a pinned Hub revision.",
    ),
    ("quit", "Quit", "Leave the MAI workbench."),
]

FORM_FIELDS: dict[str, list[Field]] = {
    "build": [
        ("spec", "Build spec", "text", "JSON spec containing dataset design and samples."),
        (
            "selected_groups",
            "Groups to build",
            "local_groups",
            "Choose exact groups from the local group catalog.",
        ),
        ("output", "Package", "text", "Local Hugging Face package directory."),
        (
            "cache",
            "Automatic cache",
            "text",
            "Reusable downloads and generated outputs; populated automatically.",
        ),
        ("force", "Replace", "toggle", "Replace an existing recognized package."),
        ("dry_run", "Dry run", "toggle", "Validate and summarize without writing."),
    ],
    "init": [
        ("spec", "Build spec", "text", "Path for the new empty build specification."),
        ("force", "Replace", "toggle", "Replace an existing specification."),
    ],
    "validate": [
        ("package", "Package", "text", "Local package containing dataset.json."),
    ],
    "publish": [
        ("package", "Package", "text", "Validated local package to upload."),
        ("repo_id", "Hub repository", "text", "Hugging Face dataset ID in OWNER/NAME form."),
        ("revision", "Branch", "text", "Target Hub branch, normally main."),
        ("tag", "Release tag", "text", "Optional immutable release tag created after upload."),
        ("private", "Private", "toggle", "Create a private dataset repository when new."),
    ],
    "pull": [
        ("repo_id", "Hub repository", "text", "Hugging Face dataset ID in OWNER/NAME form."),
        ("revision", "Revision", "text", "Commit SHA or release tag; avoid mutable main."),
        ("output", "Output", "text", "Destination for the verified local subset."),
        (
            "selected_groups",
            "Groups",
            "remote_groups",
            "Load the remote group index and choose exact groups.",
        ),
        ("force", "Replace", "toggle", "Replace an existing recognized package."),
    ],
}

FORM_TITLES = {
    "build": "Dataset · Build",
    "init": "Dataset · Initialize",
    "validate": "Dataset · Validate",
    "publish": "Dataset · Publish",
    "pull": "Dataset · Download",
}


def safe_add(
    window: curses.window,
    y: int,
    x: int,
    text: str,
    style: int = 0,
) -> None:
    height, width = window.getmaxyx()
    if y < 0 or y >= height or x < 0 or x >= width:
        return
    try:
        window.addnstr(y, x, text, max(0, width - x - 1), style)
    except curses.error:
        pass


def horizontal_rule(window: curses.window, y: int, style: int = 0) -> None:
    _, width = window.getmaxyx()
    safe_add(window, y, 1, "─" * max(0, width - 2), style)


def selected_style() -> int:
    if curses.has_colors():
        style = curses.color_pair(2)
        if getattr(curses, "COLORS", 0) < 256:
            style |= curses.A_DIM
        return style
    return curses.A_REVERSE


def initialize_colors() -> None:
    if not curses.has_colors():
        return
    curses.start_color()
    try:
        curses.use_default_colors()
    except curses.error:
        pass
    selected_foreground = (
        EXTENDED_DARK_FOREGROUND
        if getattr(curses, "COLORS", 0) >= 256
        else curses.COLOR_BLACK
    )
    try:
        curses.init_pair(2, selected_foreground, curses.COLOR_MAGENTA)
    except (curses.error, ValueError):
        pass
    for pair, foreground in (
        (1, curses.COLOR_CYAN),
        (3, curses.COLOR_GREEN),
        (4, curses.COLOR_YELLOW),
        (5, curses.COLOR_RED),
        (6, curses.COLOR_MAGENTA),
    ):
        try:
            curses.init_pair(pair, foreground, -1)
        except (curses.error, ValueError):
            pass


def draw_header(stdscr: curses.window, section: str, repository: Path) -> None:
    safe_add(stdscr, 1, 3, "MAI", curses.A_BOLD | curses.color_pair(1))
    safe_add(stdscr, 1, 8, "RESEARCH WORKBENCH", curses.A_BOLD)
    safe_add(stdscr, 2, 3, section, curses.color_pair(6))
    _, width = stdscr.getmaxyx()
    repo_text = str(repository)
    safe_add(
        stdscr,
        1,
        max(3, width - len(repo_text) - 3),
        repo_text,
        curses.A_DIM,
    )
    horizontal_rule(stdscr, 3, curses.color_pair(1))


def edit_value(
    stdscr: curses.window,
    label: str,
    initial: str,
    validator: Callable[[str], str | None] | None = None,
) -> str:
    height, width = stdscr.getmaxyx()
    box_width = min(max(48, len(initial) + 8), max(20, width - 8))
    popup = curses.newwin(7, box_width, (height - 7) // 2, (width - box_width) // 2)
    popup.keypad(True)
    value = list(initial)
    cursor = len(value)
    message = ""
    try:
        curses.curs_set(1)
    except curses.error:
        pass
    while True:
        popup.erase()
        popup.box()
        safe_add(popup, 1, 2, label, curses.A_BOLD | curses.color_pair(1))
        viewport = max(1, box_width - 6)
        offset = max(0, cursor - viewport + 1)
        safe_add(
            popup,
            3,
            2,
            "".join(value[offset : offset + viewport]),
            curses.A_UNDERLINE,
        )
        safe_add(
            popup,
            5,
            2,
            message or "Enter save · Esc cancel · Ctrl-U clear",
            curses.color_pair(5) if message else curses.A_DIM,
        )
        try:
            popup.move(3, 2 + cursor - offset)
        except curses.error:
            pass
        popup.refresh()
        key = popup.getch()
        message = ""
        if key in (10, 13, curses.KEY_ENTER):
            candidate = "".join(value)
            error = validator(candidate) if validator else None
            if error:
                message = error
                continue
            break
        if key == 27:
            value = list(initial)
            break
        if key == curses.KEY_LEFT:
            cursor = max(0, cursor - 1)
        elif key == curses.KEY_RIGHT:
            cursor = min(len(value), cursor + 1)
        elif key in (curses.KEY_HOME, 1):
            cursor = 0
        elif key in (curses.KEY_END, 5):
            cursor = len(value)
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            if cursor:
                del value[cursor - 1]
                cursor -= 1
        elif key == curses.KEY_DC and cursor < len(value):
            del value[cursor]
        elif key == 21:
            value.clear()
            cursor = 0
        elif 32 <= key <= 126:
            value.insert(cursor, chr(key))
            cursor += 1
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    return "".join(value)


def dashboard(stdscr: curses.window, repository: Path) -> str:
    selected = 0
    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        if height < 22 or width < 80:
            safe_add(stdscr, 1, 2, "Terminal must be at least 80×22.", curses.A_BOLD)
            stdscr.refresh()
            stdscr.getch()
            continue
        draw_header(stdscr, "Operations", repository)
        safe_add(stdscr, 5, 3, "Choose an operation", curses.A_BOLD)
        for index, (_, title, _) in enumerate(OPERATIONS):
            y = 7 + index
            active = index == selected
            marker = "›" if active else " "
            safe_add(
                stdscr,
                y,
                4,
                f" {marker} {title} ",
                selected_style() if active else curses.A_BOLD,
            )
        horizontal_rule(stdscr, 14)
        safe_add(stdscr, 16, 4, OPERATIONS[selected][2], curses.A_DIM)
        safe_add(stdscr, height - 2, 3, "↑/↓ navigate · Enter open · q quit", curses.A_DIM)
        stdscr.refresh()
        key = stdscr.getch()
        if key in (ord("q"), ord("Q"), 27):
            return "quit"
        if key in (curses.KEY_UP, ord("k")):
            selected = (selected - 1) % len(OPERATIONS)
        elif key in (curses.KEY_DOWN, ord("j")):
            selected = (selected + 1) % len(OPERATIONS)
        elif key in (10, 13, curses.KEY_ENTER):
            return OPERATIONS[selected][0]


def field_value(form: Any, name: str) -> str:
    value = getattr(form, name)
    if isinstance(value, bool):
        return "ON" if value else "OFF"
    if name == "selected_groups":
        if not value:
            return "Choose groups…"
        return f"{len(value)} selected · {', '.join(value)}"
    return value or "—"


def choose_groups(
    stdscr: curses.window,
    groups: list[dict[str, Any]],
    original: list[str],
) -> list[str]:
    group_ids = [group["semantic_group_id"] for group in groups]
    details = {group["semantic_group_id"]: group for group in groups}
    chosen = set(original) & set(group_ids)
    query = ""
    cursor = 0
    offset = 0
    screen_height, screen_width = stdscr.getmaxyx()
    popup_height = min(20, screen_height - 6)
    popup_width = min(84, screen_width - 12)
    popup = curses.newwin(
        popup_height,
        popup_width,
        (screen_height - popup_height) // 2,
        (screen_width - popup_width) // 2,
    )
    popup.keypad(True)
    while True:
        visible = [
            group_id
            for group_id in group_ids
            if query.casefold() in group_id.casefold()
            or query.casefold()
            in str(details[group_id].get("content_category", "")).casefold()
        ]
        cursor = min(cursor, max(0, len(visible) - 1))
        popup.erase()
        popup.box()
        safe_add(
            popup,
            1,
            2,
            "Choose semantic groups",
            curses.A_BOLD | curses.color_pair(1),
        )
        safe_add(popup, 2, 2, f"{len(chosen)} of {len(group_ids)} selected")
        if query:
            safe_add(popup, 3, 2, f"Search: {query}", curses.color_pair(4))
        horizontal_rule(popup, 4)
        list_top = 5
        page_size = max(1, popup_height - list_top - 3)
        if cursor < offset:
            offset = cursor
        if cursor >= offset + page_size:
            offset = cursor - page_size + 1
        for row, group_id in enumerate(visible[offset : offset + page_size]):
            index = offset + row
            group = details[group_id]
            marker = "✓" if group_id in chosen else " "
            text = (
                f" [{marker}] {group_id} · "
                f"{group.get('content_category', '')} · "
                f"{group.get('sample_count', len(group.get('samples', [])))} "
                f"{group.get('sample_count_label', 'samples')} "
            )
            safe_add(
                popup,
                list_top + row,
                2,
                text,
                selected_style()
                if index == cursor
                else curses.color_pair(3) if group_id in chosen else 0,
            )
        if not visible:
            safe_add(popup, list_top, 2, "No groups match.", curses.A_DIM)
        safe_add(
            popup,
            popup_height - 2,
            2,
            "↑/↓ · Space · a all · n clear · / search · Enter save · Esc",
            curses.A_DIM,
        )
        popup.refresh()
        key = popup.getch()
        if key == 27:
            return original
        if key in (10, 13, curses.KEY_ENTER):
            return [group_id for group_id in group_ids if group_id in chosen]
        if key in (curses.KEY_UP, ord("k")) and visible:
            cursor = (cursor - 1) % len(visible)
        elif key in (curses.KEY_DOWN, ord("j")) and visible:
            cursor = (cursor + 1) % len(visible)
        elif key == ord(" ") and visible:
            group_id = visible[cursor]
            if group_id in chosen:
                chosen.remove(group_id)
            else:
                chosen.add(group_id)
        elif key in (ord("a"), ord("A")):
            chosen.update(visible)
        elif key in (ord("n"), ord("N")):
            chosen.clear()
        elif key == ord("/"):
            query = edit_value(stdscr, "Search groups", query)
            cursor = 0
            offset = 0


def confirm(stdscr: curses.window, prompt: str, detail: str | None = None) -> bool:
    height, width = stdscr.getmaxyx()
    box_width = min(max(72, len(prompt) + 6), width - 6)
    lines = (
        textwrap.wrap(detail, max(1, box_width - 4), break_long_words=True)
        if detail
        else []
    )
    lines = lines[: max(0, height - 10)]
    box_height = 6 + len(lines)
    popup = curses.newwin(
        box_height,
        box_width,
        (height - box_height) // 2,
        (width - box_width) // 2,
    )
    popup.box()
    safe_add(popup, 1, 2, "Confirm", curses.A_BOLD | curses.color_pair(4))
    safe_add(popup, 2, 2, prompt)
    for index, line in enumerate(lines):
        safe_add(popup, 4 + index, 2, line, curses.color_pair(3))
    safe_add(popup, box_height - 2, 2, "y confirm · any other key cancel", curses.A_DIM)
    popup.refresh()
    return popup.getch() in (ord("y"), ord("Y"))


def notice(stdscr: curses.window, title: str, detail: str) -> None:
    height, width = stdscr.getmaxyx()
    box_width = min(max(56, len(title) + 6), width - 6)
    lines = textwrap.wrap(detail, max(1, box_width - 4), break_long_words=True)
    lines = lines[: max(1, height - 9)]
    box_height = 5 + len(lines)
    popup = curses.newwin(
        box_height,
        box_width,
        (height - box_height) // 2,
        (width - box_width) // 2,
    )
    popup.box()
    safe_add(popup, 1, 2, title, curses.A_BOLD | curses.color_pair(4))
    for index, line in enumerate(lines):
        safe_add(popup, 2 + index, 2, line)
    safe_add(popup, box_height - 2, 2, "Press any key to return", curses.A_DIM)
    popup.refresh()
    popup.getch()


def output_reader(process: subprocess.Popen[str], output: Queue[str]) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        output.put(line.rstrip())


def run_process(stdscr: curses.window, repository: Path, command: list[str]) -> int:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
    )
    queue: Queue[str] = Queue()
    threading.Thread(
        target=output_reader,
        args=(process, queue),
        daemon=True,
    ).start()
    log: deque[str] = deque(maxlen=2000)
    stdscr.timeout(100)
    cancelled = False
    while process.poll() is None or not queue.empty():
        while True:
            try:
                log.append(queue.get_nowait())
            except Empty:
                break
        stdscr.erase()
        height, _ = stdscr.getmaxyx()
        draw_header(stdscr, "Operation · Running", repository)
        safe_add(
            stdscr,
            5,
            3,
            "RUNNING" if process.poll() is None else "FINISHING",
            curses.A_BOLD | curses.color_pair(4),
        )
        for offset, line in enumerate(list(log)[-max(1, height - 10) :]):
            safe_add(stdscr, 7 + offset, 3, line)
        safe_add(
            stdscr,
            height - 2,
            3,
            "q cancel operation" if process.poll() is None else "collecting output…",
            curses.A_DIM,
        )
        stdscr.refresh()
        key = stdscr.getch()
        if key in (ord("q"), ord("Q")) and process.poll() is None:
            if confirm(stdscr, "Terminate the running operation?"):
                process.terminate()
                cancelled = True
    return_code = process.wait()
    stdscr.timeout(-1)
    while True:
        try:
            log.append(queue.get_nowait())
        except Empty:
            break
    stdscr.erase()
    height, _ = stdscr.getmaxyx()
    draw_header(stdscr, "Operation · Result", repository)
    if return_code == 0 and not cancelled:
        status, style = "COMPLETED", curses.color_pair(3)
    elif cancelled:
        status, style = "CANCELLED", curses.color_pair(4)
    else:
        status, style = f"FAILED · exit {return_code}", curses.color_pair(5)
    safe_add(stdscr, 5, 3, status, curses.A_BOLD | style)
    for offset, line in enumerate(list(log)[-max(1, height - 10) :]):
        safe_add(stdscr, 7 + offset, 3, line)
    safe_add(stdscr, height - 2, 3, "Press any key to return", curses.A_DIM)
    stdscr.refresh()
    stdscr.getch()
    return return_code


def draw_form(
    stdscr: curses.window,
    repository: Path,
    operation: str,
    form: Any,
    selected: int,
    message: str,
) -> None:
    fields = FORM_FIELDS[operation]
    stdscr.erase()
    height, _ = stdscr.getmaxyx()
    draw_header(stdscr, FORM_TITLES[operation], repository)
    descriptions = {key: description for key, _, description in OPERATIONS}
    safe_add(stdscr, 5, 3, descriptions[operation], curses.A_BOLD)
    label_width = 18
    value_x = 4 + label_width
    for index, (name, label, _, _) in enumerate(fields):
        y = 7 + index
        active = index == selected
        safe_add(
            stdscr,
            y,
            4,
            f"{label:<{label_width}}",
            curses.A_BOLD if active else 0,
        )
        safe_add(
            stdscr,
            y,
            value_x,
            f" {field_value(form, name)} ",
            selected_style() if active else curses.color_pair(1),
        )
    action_y = 7 + len(fields) + 1
    run_selected = selected == len(fields)
    back_selected = selected == len(fields) + 1
    safe_add(
        stdscr,
        action_y,
        4,
        " Run operation ",
        selected_style()
        if run_selected
        else curses.color_pair(3) | curses.A_BOLD,
    )
    safe_add(
        stdscr,
        action_y,
        23,
        " Back ",
        selected_style() if back_selected else curses.A_BOLD,
    )
    horizontal_rule(stdscr, action_y + 1)
    errors = form.errors(repository)
    if message:
        detail, style = message, curses.color_pair(4)
    elif errors:
        detail, style = "Validation: " + " ".join(errors), curses.color_pair(5)
    elif selected < len(fields):
        detail, style = fields[selected][3], curses.A_DIM
    elif run_selected:
        detail, style = "Review the command, then confirm execution.", curses.A_DIM
    else:
        detail, style = "Return to operations.", curses.A_DIM
    safe_add(stdscr, action_y + 2, 4, detail, style)
    preview = "Command: " + command_preview(form, repository)
    safe_add(
        stdscr,
        height - 3,
        3,
        preview,
        curses.A_DIM if errors else curses.color_pair(3),
    )
    safe_add(
        stdscr,
        height - 1,
        3,
        "↑/↓ navigate · Enter edit/toggle · Esc back",
        curses.A_DIM,
    )
    stdscr.refresh()


def create_form(operation: str) -> Any:
    return {
        "build": BuildForm,
        "init": InitForm,
        "validate": ValidateForm,
        "publish": PublishForm,
        "pull": PullForm,
    }[operation]()


def operation_form(
    stdscr: curses.window,
    repository: Path,
    operation: str,
) -> None:
    form = create_form(operation)
    fields = FORM_FIELDS[operation]
    selected = 0
    message = ""
    total = len(fields) + 2
    while True:
        height, width = stdscr.getmaxyx()
        if height < 24 or width < 80:
            stdscr.erase()
            safe_add(stdscr, 1, 2, "Form needs a terminal of at least 80×24.")
            stdscr.refresh()
            if stdscr.getch() == 27:
                return
            continue
        draw_form(stdscr, repository, operation, form, selected, message)
        key = stdscr.getch()
        message = ""
        if key == 27:
            return
        if key in (curses.KEY_UP, ord("k")):
            selected = (selected - 1) % total
        elif key in (curses.KEY_DOWN, ord("j"), 9):
            selected = (selected + 1) % total
        elif key in (10, 13, curses.KEY_ENTER, ord(" ")):
            if selected < len(fields):
                name, label, kind, _ = fields[selected]
                if kind == "toggle":
                    setattr(form, name, not getattr(form, name))
                elif kind in {"local_groups", "remote_groups"}:
                    if kind == "local_groups":
                        source = resolve(repository, form.spec)
                        if not source.is_file():
                            notice(
                                stdscr,
                                "Build spec not found",
                                (
                                    f"{source} does not exist. Create it with "
                                    "Initialize build spec before choosing groups."
                                ),
                            )
                            continue
                        loading = str(source)
                    else:
                        if not form.repo_id or not form.revision:
                            message = "Enter a Hub repository and revision first."
                            continue
                        loading = f"{form.repo_id}@{form.revision}"
                    stdscr.erase()
                    draw_header(stdscr, "Dataset · Loading group index", repository)
                    safe_add(stdscr, 5, 3, loading)
                    stdscr.refresh()
                    try:
                        groups = (
                            list_spec_groups(source)
                            if kind == "local_groups"
                            else list_remote_groups(form.repo_id, form.revision)
                        )
                    except PipelineError as error:
                        notice(stdscr, "Cannot load groups", str(error))
                    else:
                        if not groups:
                            notice(
                                stdscr,
                                "No groups in build spec",
                                (
                                    "The spec exists but its group list is empty. "
                                    "Add semantic groups, then open this selector again."
                                ),
                            )
                        else:
                            form.selected_groups = choose_groups(
                                stdscr,
                                groups,
                                form.selected_groups,
                            )
                else:
                    setattr(
                        form,
                        name,
                        edit_value(stdscr, label, str(getattr(form, name))),
                    )
            elif selected == len(fields):
                errors = form.errors(repository)
                if errors:
                    message = " ".join(errors)
                    continue
                command = form.argv(repository)
                if confirm(
                    stdscr,
                    "Run this operation?",
                    command_preview(form, repository),
                ):
                    run_process(stdscr, repository, command)
            else:
                return


def application(stdscr: curses.window, repository: Path) -> int:
    stdscr.keypad(True)
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    initialize_colors()
    while True:
        operation = dashboard(stdscr, repository)
        if operation == "quit":
            return 0
        operation_form(stdscr, repository, operation)


def run(repository: Path) -> int:
    try:
        return curses.wrapper(application, repository)
    except KeyboardInterrupt:
        return 130
