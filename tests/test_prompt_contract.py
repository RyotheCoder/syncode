"""Hop dong prompt: rut gon/to them khong duoc lam mat quy tac nao.

Moi lan sua roles.py hay tool description ma lam mat 1 cum bat buoc duoi day
-> test rot. Day la bang chung "rut gon nhung khong doi noi dung".
"""

from syncode.swarm import roles
from syncode.tools.registry import BUILTIN_TOOLS

# Tong ky tu toan bo system prompt (cu: 7476) — khoa muc rut gon da dat.
PROMPT_BUDGET_CHARS = 7000

ROLE_MUST = {
    "PLANNER_SYSTEM": [
        "MỘT mảng JSON", "specialist", "ĐÚNG 1 subtask", "2–5",
        "ask_user/qa_user", "reviewer", "Analyze...", "Step...",
    ],
    "WORKER_SYSTEM": [
        "BẮT BUỘC gọi tool", "ask_user/qa_user", "KHÔNG nhận được trả lời",
        "KẾT QUẢ cuối", "Q -> A", "tiếng Việt", "Analyze user input", "identify core task",
        "Step...", "Mã triển khai", "Tổng kết", "Subtask",
        "planner/worker/swarm/subtask", "read_file", "edit_file",
        "web_search", "calculator", "≤ 8 dòng", "≤ 15 dòng", "≤ 5 dòng",
        "đáp án đã có sẵn", "TỐI ĐA 1 lần", "không hỏi dồn",
        "sửa args thử lại 1 lần",
    ],
    "CODER_SYSTEM": [
        "BẮT BUỘC gọi tool", "ĐẦY ĐỦ", "≤ 5 dòng", "Analyze user input",
        "Tổng kết",
    ],
    "REVIEWER_SYSTEM": [
        "ĐẠT, không thấy lỗi", "run_command", "file:dòng",
    ],
    "RESEARCHER_SYSTEM": [
        "gạch đầu dòng", "file:dòng / URL",
    ],
    "CRITIC_SYSTEM": [
        "APPROVED", "REVISE", "issues", "guidance", "secret",
        "nghi ngờ không chắc → APPROVED",
    ],
    "REFINER_SYSTEM": [
        "Chỉ sửa issue", "bản viết lại",
    ],
    "JUDGE_SYSTEM": [
        "ĐÃ HỎI QUA TOOL", "KHÔNG hỏi lại bằng text", "KẾT LUẬN",
        "Analyze user input", "Mã triển khai", "≤ 5 dòng", "tiếng Việt",
    ],
    "QUICK_SYSTEM": [
        "KHÔNG dùng tool", "1–3 câu", "≤ 15 dòng",
    ],
}


def test_role_prompts_keep_every_rule():
    for name, phrases in ROLE_MUST.items():
        prompt = getattr(roles, name)
        assert prompt and prompt.strip(), f"{name} rong"
        for ph in phrases:
            assert ph in prompt, f"{name} mat quy tac: {ph!r}"


def test_prompt_budget():
    total = sum(
        len(getattr(roles, name))
        for name in ROLE_MUST
    )
    assert total <= PROMPT_BUDGET_CHARS, f"prompt phinh {total} > budget {PROMPT_BUDGET_CHARS}"


def test_ask_tools_descriptions():
    descs = {t.name: t.description for t in BUILTIN_TOOLS}
    assert "KHÔNG nhận được trả lời" in descs["ask_user"]
    assert "BẮT BUỘC" in descs["ask_user"]
    assert "Bí danh của ask_user" in descs["qa_user"]
    assert "bắt buộc gọi" in descs["qa_user"]
