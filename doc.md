# Quizzez App - Core Workflows & Architecture

Tài liệu này giải thích các luồng hoạt động (workflow) cốt lõi của **Quizzez App**, dựa trên thiết kế **Clean Architecture**, **MVVM** và **Local-first Sync**.

---

## 1. Kiến trúc tổng thể (Clean Architecture + MVVM)

Dự án được phân tách thành 3 layer (tầng) hoàn toàn tách biệt, ngăn chặn việc mã nguồn bị liên kết cứng (tight coupling):

```text
+-------------------------------------------------------------+
|                     1. UI LAYER (Presentation)              |
|  +--------------------+       +--------------------------+  |
|  | Jetpack Compose  <--------->      ViewModels          |  |
|  | (Stateless)        |       | (Quản lý State)          |  |
|  +--------------------+       +--------------------------+  |
+-------------------------------------------------------------+
               |  (Gửi Event)                    ^
               v                                 | (Emit StateFlow)
+-------------------------------------------------------------+
|                     2. DOMAIN LAYER (Core)                  |
|                                                             |
|   +-------------------+       +-----------------------+     |
|   | Repository        |       | Domain Models         |     |
|   | Interfaces        |       | (Quiz, User...)       |     |
|   +-------------------+       +-----------------------+     |
|                          (Utils)                            |
+-------------------------------------------------------------+
               ^
               | (Implements)
               v
+-------------------------------------------------------------+
|                     3. DATA LAYER                           |
|                                                             |
|           [ Repository Implementations ]                    |
|                /                  \                         |
|   (Ghi luôn)  /                    \ (Sync & Fallback)      |
|              v                      v                       |
|  +----------------+        +--------------------------+     |
|  | Room Database  |        | Firebase Firestore (API) |     |
|  | (Local)        |        | (Remote)                 |     |
|  +----------------+        +--------------------------+     |
+-------------------------------------------------------------+
```

### Nguyên tắc hoạt động
- **Domain Layer:** Nằm ở trung tâm kiến trúc, bao gồm thuần mã Kotlin không có các thư viện của Android framework hay Firebase. Thành phần này chứa logic cốt lõi như rules của trò chơi (tối đa câu hỏi, quy tắc chấm điểm `ScoreCalculator`).
- **Data Layer:** Phụ trách trích xuất và biến đổi nguồn dữ liệu. Gọi dữ liệu từ nội bộ thiết bị (Room SQLite) hoặc giao tiếp với mạng ngoài (Firebase). Nhiệm vụ của nó là fetch dữ liệu, ép kiểu (từ `Entity/Dto` sang `Domain Model`) và cung cấp lại cho interface.
- **UI Layer:** Chỉ tập trung cho việc vẽ giao diện (Jetpack Compose). Toàn bộ Logic được đẩy hết sang cho ViewModel.

---

## 2. Luồng Trạng Thái & Giao Diện (Unidirectional Data Flow)

Dự án sử dụng luồng dữ liệu một chiều (UDF). Mọi thay đổi đều được kiểm soát bởi duy nhất một nơi là **ViewModel**.

```text
                    [ COMPOSE SCREEN ]
                            |
                            | 1. Người dùng thao tác (Click, Nhập form)
                            | 2. Gọi logic: onEvent(Event)
                            v
+---------------------------------------------------------+
|                      [ VIEWMODEL ]                      |
|                                                         |
|  1. Nhận Event từ giao diện                             |
|  2. Đẩy thao tác cho Repository (Room/Firebase API)     |
|  3. Chờ kết quả trả về (Result/Flow)                    |
|  4. Cập nhật lại biến lõi _uiState với dữ liệu mới      |
+---------------------------------------------------------+
                            |
                            | 3. Phát loa tín hiệu mới qua StateFlow
                            | 4. UI lặp lại bước vẽ (Re-render)
                            v
                    [ COMPOSE SCREEN ]
```

### Nguyên tắc hoạt động
1. Compose UI luôn ở trạng thái tĩnh (Stateless), không được lưu trữ dữ liệu cục bộ làm thay đổi giao diện.
2. Thao tác người dùng (Click chuột, Nhập form) gọi thành `Event` truyền thông qua hàm `onEvent()` tới ViewModel.
3. Khi Logic hay API thay đổi trả về kết quả thành công, ViewModel cập nhật lại cái lõi trạng thái (StateFlow).
4. Jetpack Compose tự động nghe sự kiện thay đổi dữ liệu và thực hiện render (vẽ lại) giao diện cần thiết rất nhẹ nhàng.

---

## 3. Workflow Đồng Bộ Dữ Liệu (Local-first & Cloud Sync)

Để ứng dụng không bao giờ bị đứng máy khi mạng yếu hay bị gián đoạn, Quizzez sử dụng cơ chế bảo hiểm lưu trữ cục bộ gọi là **Local-first thiết kế cùng Background Sync**.

```text
[ USER LƯU DỮ LIỆU ]  --->  VIEWMODEL  --->  [ REPOSITORY IMPLEMENTATIONS ]
                                                            |
                                                            v
               +--------------------------------------------------------------+
               |                  1. THAO TÁC LOCAL (TỨC THÌ)                 |
               |   -> Ghi thẳng vào: Room Database (Gắn cờ: PENDING)          |
               |   -> Thêm công việc vào: PendingSyncEntity (Hàng đợi)        |
               +--------------------------------------------------------------+
                            /                                 \
      [ BÁO THÀNH CÔNG VỚI APP ]                            [ KÍCH HOẠT WORKER CHẠY NGẦM ]
              /                                                       \
+---------------------------+                        +-----------------------------------------+
| UI cập nhật ngay lập tức  |                        | Sync Manager bắt đầu đọc danh sách nợ   |
| Không hề có vòng Quay Đợi |                        |            (Có mạng không?)             |
+---------------------------+                        +-----------------------------------------+
                                                              /                   \
                                                          [ CÓ MẠNG ]        [ RỚT MẠNG / LỖI ]
                                                              /                       \
                                         +---------------------------+       +---------------------------+
                                         | - Đẩy ngược lên Firebase  |       | - Tăng retryCount (+1)    |
                                         | - Xóa sổ nợ Pending Queue |       | - Chờ lần sau thử lại     |
                                         | - Status chuyển qua SYNCED|       |                           |
                                         +---------------------------+       +---------------------------+
```

### Nguyên tắc hoạt động
- **Ưu tiên nội bộ (Local-first):** Thiết bị lưu data vào file SQLite (Room Database) và báo với hệ thống là đã lưu xong, nên UI của app sẽ chạy liền mạch không có vòng xoay Loading (Spinner). Bản ghi sẽ bị dán cái mác "chưa đồng bộ" (`syncStatus = PENDING`).
- **Hàng đợi làm việc (Pending Sync Queue):** Một bảng riêng rẽ (`PendingSyncEntity`) quản lý dữ liệu nào của user đang mắc nợ trên máy tính mà chưa ném cho Backend.
- **Rảnh rỗi sinh nông nổi (Background Push):** Đằng sau hậu trường, `SyncManager` mở danh sách nợ trên. Chờ tín hiệu kết nối mạng hợp lệ là tóm từng cái nợ đồng bộ âm thầm qua API (`Push`). Nếu ném API tạch, file nợ tăng điểm `retryCount` chờ thử lại lần sau. Lên Cloud thành công thì gạch sổ nợ đi (`Remove`).
