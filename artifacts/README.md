# Model Artifacts

## SUN397 web bundles đã xác minh và tích hợp

`deployment/sun397/tacs_visual/` và `deployment/sun397/mtacs_full_both/` chứa hai bundle L/16–384, K=1, pool 3.573 ảnh. Toàn bộ file trong manifest đã được đối chiếu SHA-256 trước khi di chuyển; giữ nguyên model/cache/pool/source. `deployment/sun397/relocation.json` lưu đường dẫn cũ và mới.

Web tại `apps/sun397_demo` dùng trực tiếp hai bundle này và tự nạp model vào worker khi khởi động. Đây là các bundle đã xác minh, tách biệt với checkpoint nhập cũ bên dưới. Hướng dẫn: `apps/sun397_demo/README.md`.

## Các checkpoint nhập cũ

`checkpoints/imported_unverified/` preserves both original sources (`root/` and `model_TACS/`).
Files with the same basename have NOT been silently deduplicated, converted or loaded.
That folder includes the originally colocated selector cache and candidate feature bank as well as checkpoints;
keeping the original bundle together preserves provenance.

Architecture, class-index order, candidate pool identity, backbone and checkpoint schema must be verified before inference.
No imported weight was connected to the web app during organization.

Empty `feature_banks/`, `text_embeddings/` and `captioners/` folders are for newly verified exports.
Keep large assets out of Git; record their location and hashes in an experiment manifest.
