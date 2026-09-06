# Đóng góp cho Syncode

Cảm ơn bạn đã quan tâm! Repo này ưu tiên: **đáp án gọn đúng trọng tâm**,
**không rò reasoning vào output**, **mọi behavior có test**.

## Chuẩn bị

```bash
pip install -e ".[dev]"
pytest -q   # phải xanh hết trước khi gửi PR
```

## Quy ước code

- UI hiển thị **tiếng Anh + ASCII** (giữ các ký tự `✓ ✗ · │ ○ ▸`);
  prompt và comment trong code giữ như hiện tại, không Việt hoá ồ ạt.
- Không emoji trong code (trừ khi user yêu cầu).
- Không bao giờ commit secret, API key, file trong `~/.syncode/`
  (history cá nhân không thuộc repo).
- Prompt trong `swarm/roles.py` và mô tả tool là **hợp đồng hành vi**:
  sửa/xoá luật nào cũng phải cập nhật `tests/test_prompt_contract.py`
  (test sẽ rot nếu mất luật — đó là chủ ý).

## Thêm slash command

1. Viết handler `_cmd_ten(ctx, args) -> bool` trong `syncode/commands/slash.py`
   (True = thoát app).
2. Đăng ký vào `COMMANDS`: `"ten": (_cmd_ten, "[args]", "Mo ta ngan")`.
3. Thêm test trong `tests/test_smoke.py` hoặc file test riêng.

## Thêm tool builtin

1. Viết handler `_tool_ten(args) -> str` trong `syncode/tools/registry.py`
   (không raise — trả chuỗi lỗi, cắt output dài).
2. Thêm `Tool(name=..., description=..., parameters=..., handler=...)` vào
   `BUILTIN_TOOLS`. Mô tả phải nói rõ **khi nào DÙNG và khi nào KHÔNG dùng**.
3. Thêm test trong `tests/test_tools.py`.

## Thêm/sửa role trong swarm

1. Sửa `swarm/roles.py`, giữ nguyên tắc "mỗi agent 1 vai".
2. Cập nhật `ROLE_PROMPTS` + `tests/test_prompt_contract.py`.
3. Chạy full suite + 1 lượt live kiểm chứng nếu đổi behavior.

## Quy trình PR

1. Fork → nhánh mới từ `main` (`git checkout -b feat/ten-tinh-nang`).
2. Code + test xanh (`pytest -q`), mô tả rõ behavior đổi gì.
3. Mở PR vào `Ryothecoder/syncode` — maintainer review rồi merge.

## Báo lỗi

Mở issue với: câu hỏi gây lỗi, output của `/.debug`, model đang dùng
(`/.model`), mode (`/.mode`). Log/chat history cá nhân nhớ xoá key trước khi dán.
