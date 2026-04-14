# VeriFact App

This directory contains the Flutter client application for VeriFact.

It is the user-facing layer that:

- captures user input (typed claims/questions and image content),
- performs on-device OCR,
- sends requests to retrieval/QA/verifier backend endpoints,
- renders evidence-driven outputs,
- persists user history locally.

## 1) Why This App Exists

The App is built to make medical fact-checking workflows practical for non-technical users. It combines:

- low-friction input paths (typing, camera capture, gallery upload),
- transparent outputs (confidence, source links, score context),
- persistent local recall (history previews and reruns).

The design goal is to keep the UI simple while backend and model complexity remain server-side.

## 2) Tech Stack Snapshot

- **Framework**: Flutter / Dart SDK `^3.10.1`
- **State Management**: Riverpod 3 (`flutter_riverpod`, `riverpod_annotation`, generated providers)
- **Networking**: Dio
- **OCR**: `google_mlkit_text_recognition`
- **Persistence**: Hive + SharedPreferences
- **Image Capture/Selection**: `camera` + `image_picker`
- **Permissions**: `permission_handler`
- **Responsive/UI Utilities**: `flutter_screenutil`, `flutter_native_splash`, `flutter_launcher_icons`

## 3) Runtime Flavors and Boot Flow

Flavor entrypoints:

- `lib/main_dev.dart`
- `lib/main_prod.dart`

Common boot sequence in `lib/main_common.dart`:

1. Initialize `FlavorConfig` (base URL + flavor identity).
2. Ensure Flutter bindings and preserve splash screen.
3. Preload available cameras.
4. Request camera/gallery permissions.
5. Initialize Hive and register `HistoryRecordAdapter`.
6. Open `history` box eagerly.
7. Lock orientation to portrait.
8. Remove splash and run app inside `ProviderScope`.

### Environment Variables

The app reads from `.env`:

- `API_BASE_URL_DEV` in dev mode
- `API_BASE_URL_PROD` in prod mode

`main_dev.dart` and `main_prod.dart` validate that required variables are present.

## 4) Application Architecture

The app uses a layered architecture:

1. **UI Layer** (`lib/screens`, `lib/widgets`)
2. **State Layer** (`lib/utils/notifiers`)
3. **Repository Layer** (`lib/utils/repositories`)
4. **Controller Layer** (`lib/controllers`)
5. **Transport + Storage Services** (`lib/utils/services`, `lib/services`)
6. **Typed Models** (`lib/models`)

This separation keeps UI concerns isolated from API transport and business flow, making the app easier to evolve and test.

## 5) Screen System and User Flows

Primary screens in `lib/screens/`:

- `splash_screen.dart`
- `home_screen.dart`
- `verifier_result_screen.dart`
- `history_screen.dart`
- `history_preview_screen.dart`
- `ocr_page.dart`
- `settings_screen.dart`
- `about_screen.dart`
- `response_test_screen.dart`

### Home Modes

`HomeScreen` supports three search modes via `HomeSearchNotifier`:

- `verifier`
- `qa`
- `doc`

Behavior:

- Verifier mode navigates to `VerifierResultScreen` with the claim.
- QA mode fetches `/qa/answer`, renders parsed answer, stores result in history.
- Doc mode fetches `/retrieval/search`, renders ranked passages, stores result in history.

### Quick Actions

Home quick actions provide:

- QA mode selection
- Doc Search mode selection
- Camera capture (`CameraService.openCameraAndShow`)
- Gallery upload (`ImagePickerService.pickFromGallery` + preview path)

## 6) OCR Subsystem

OCR stack is implemented by `OcrService` and `ocr_provider.dart`.

Key responsibilities:

- permission-aware image selection/capture,
- robust image file validation,
- OCR processing via ML Kit `TextRecognizer`,
- extraction of text blocks with bounding boxes,
- optional fine-grained text selection overlays in `ocr_page.dart`.

Data model:

- `TextBlock` contains text, absolute bounding box, source image dimensions, and normalized coordinates for UI rendering.

Why this exists:

- to support image-first fact-checking where users start with screenshots/photos,
- to make extracted text inspectable before verification call submission.

## 7) Networking and API Contract Layer

### URL Constants

`lib/utils/constants/url_strings.dart` defines endpoint paths:

- `/verifier/verify`
- `/retrieval/search`
- `/qa/answer`

### Dio Client Strategy

`DioClient` supports two client builders:

- `initClient()`
- `initPublicClient()`

Both support runtime base URL override from SharedPreferences key `override_base_url`; otherwise they use flavor base URL.

A logging interceptor is attached for request/response/error traceability.

### Controller Responsibilities

- `QAController`: maps request payload and parses `QAModel`
- `RetrieverController`: maps retrieval request and parses `RetrieverResponse`
- `VerifierController`: posts claim payload and parses `VerifierModel`

Controllers convert transport-level failures into typed exceptions for UI-safe handling.

### Repository + Notifier Pattern

- Repositories wrap controllers for dependency injection boundaries.
- Riverpod notifiers (`qa_notifier`, `retriever_notifier`, `verifier_notifier`) expose async state lifecycle to UI.

This keeps screens declarative and eliminates manual state plumbing.

## 8) Local Persistence and History

History is stored in Hive box `history` via `HistoryService`.

Record model (`HistoryRecord`):

- `type` (`verifier`, `qa`, `doc`)
- `query`
- `resultStatus`
- `conclusion`
- `evidence`
- `sources`
- `payload`
- `timestamp`

Features:

- grouped-by-day rendering (`Today`, `Yesterday`, weekday labels),
- type filtering in UI,
- preview reconstruction for QA/doc/verifier payloads,
- swipe-to-delete.

Why this exists:

- users need to revisit prior checks,
- allows quick audit trail of app-level fact-check interactions.

## 9) Settings and Runtime Control

`settings_screen.dart` includes:

- theme toggle (persisted using `theme_notifier` and SharedPreferences),
- About navigation,
- long-press base URL override editor.

The base URL override enables rapid environment switching during demos/testing without rebuilding flavors.

## 10) App Directory Map

```text
App/
├─ lib/
│  ├─ main_dev.dart
│  ├─ main_prod.dart
│  ├─ main_common.dart
│  ├─ screens/
│  ├─ controllers/
│  ├─ providers/
│  ├─ services/
│  ├─ utils/
│  │  ├─ constants/
│  │  ├─ notifiers/
│  │  ├─ repositories/
│  │  └─ services/
│  ├─ models/
│  └─ widgets/
├─ assets/
├─ android/
├─ ios/
└─ tools/
```

## 11) Setup and Run

From `App/`:

```bash
flutter pub get
```

Run dev flavor:

```bash
flutter run -t lib/main_dev.dart --flavor dev
```

Run prod flavor:

```bash
flutter run -t lib/main_prod.dart --flavor prod
```

Build release appbundle:

```bash
flutter build appbundle --flavor prod -t lib/main_prod.dart --release
```

## 12) Development Commands

```bash
flutter analyze
dart run build_runner build --delete-conflicting-outputs
```

## 13) Common Troubleshooting

### App fails at startup due missing API env

- Ensure `.env` exists in `App/` and includes required `API_BASE_URL_DEV` / `API_BASE_URL_PROD`.

### OCR not working

- Confirm camera/photos permissions granted.
- Check image source accessibility and file existence.

### API calls fail after endpoint migration

- Update flavor env values or long-press `Change baseURL` in Settings.
- Verify backend route availability (`/qa/answer`, `/retrieval/search`, `/verifier/verify`).

### History not appearing

- Ensure Hive initialization is successful in boot flow.
- Verify `HistoryRecordAdapter` type id compatibility if model changes are introduced.

## 14) Why This App Structure Works

- Keeps business logic out of widgets.
- Makes async API/UI state explicit and predictable.
- Supports multi-environment operation with minimal friction.
- Preserves user trust with evidence visibility and persistent history.

This App is intentionally designed as the reliable, user-facing shell over the broader VeriFact backend and model ecosystem.
