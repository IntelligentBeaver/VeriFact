# OCR Page - Before & After Comparison

## State Management

### Before (ConsumerStatefulWidget)
```dart
class _OcrScreenState extends ConsumerState<OcrScreen> {
  String _recognizedText = '';
  XFile? _imageFile;
  bool _isProcessing = false;
  File? selectedImage;

  @override
  void dispose() {
    _textRecognizer.close();
    super.dispose();
  }

  Future<void> _pickImage(ImageSource source) async {
    setState(() {
      _isProcessing = true;
      _recognizedText = '';
    });
    // ... logic
    setState(() {
      _imageFile = pickedFile;
      _recognizedText = recognizedText.text;
    });
  }
}
```

### After (ConsumerWidget + Riverpod v3)
```dart
class OcrScreen extends ConsumerWidget {
  const OcrScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final ocrState = ref.watch(ocrNotifierProvider);
    final ocrNotifier = ref.read(ocrNotifierProvider.notifier);
    
    // Reactive - no setState needed!
    return Scaffold(/* ... */);
  }
}
```

**Benefits:**
- ✅ No manual state tracking
- ✅ Reactive UI updates
- ✅ Cleaner widget rebuild
- ✅ No disposal logic needed in widget

---

## Permission Handling

### Before
```dart
Future<bool> _requestPermissionFor(ImageSource source) async {
  if (source == ImageSource.camera) {
    var status = await Permission.camera.status;
    if (status.isDenied) {
      status = await Permission.camera.request();
    }
    if (status.isPermanentlyDenied) return false;
    return status.isGranted;
  } else {
    // Manual gallery handling
    if (Platform.isIOS) {
      // iOS specific logic
    } else {
      // Android specific logic
    }
  }
  // Mixed in _pickImage logic
}
```

### After (OcrService)
```dart
Future<bool> requestPermission(ImageSource source) async {
  if (source == ImageSource.camera) {
    return _requestCameraPermission();
  } else {
    return _requestGalleryPermission();
  }
}

Future<bool> _requestCameraPermission() async {
  final status = await Permission.camera.request();
  // Centralized, clean logic
}

Future<bool> _requestGalleryPermission() async {
  if (Platform.isIOS) {
    return _requestGalleryPermissionIOS();
  } else {
    return _requestGalleryPermissionAndroid();
  }
}
```

**Benefits:**
- ✅ Separated from UI logic
- ✅ Reusable service class
- ✅ Easier to test and maintain
- ✅ Clear separation of concerns

---

## Image Picking

### Before
```dart
final pickedFile = await _picker
    .pickImage(source: source, maxWidth: 2048, imageQuality: 85)
    .timeout(
      const Duration(seconds: 20),
      onTimeout: () {
        debugPrint('[picker] pickImage timed out after 20s');
        return null;
      },
    );
```

### After (OcrService)
```dart
Future<XFile?> pickImage(ImageSource source) async {
  try {
    // Check permission first
    final hasPermissionResult = await hasPermission(source);
    if (!hasPermissionResult) {
      final permissionGranted = await requestPermission(source);
      if (!permissionGranted) {
        throw Exception('PERMISSION_DENIED');
      }
    }

    final pickedFile = await _picker.pickImage(
      source: source,
      maxWidth: 2048,
      maxHeight: 2048,
      imageQuality: 85,
    ).timeout(
      const Duration(seconds: 30),
      onTimeout: () {
        debugPrint('[OCR Picker] pickImage timed out after 30s');
        return null;
      },
    );

    // Verify file exists
    final file = File(pickedFile.path);
    if (!await file.exists()) {
      throw Exception('Selected file does not exist');
    }

    return pickedFile;
  } catch (e) {
    debugPrint('[OCR Picker] Error picking image: $e');
    rethrow;
  }
}
```

**Benefits:**
- ✅ Robust error handling
- ✅ File existence verification
- ✅ Longer timeout (30s instead of 20s)
- ✅ Permission check before picker
- ✅ Reusable across the app

---

## Error Handling

### Before
```dart
} catch (e) {
  debugPrint('Error picking image: $e');
  if (!mounted) return;
  ScaffoldMessenger.of(
    context,
  ).showSnackBar(SnackBar(content: Text('Failed to pick image: $e')));
} finally {
  setState(() => _isProcessing = false);
}

void _showPermissionDeniedDialog() {
  showErrorSnackbar('Permission Denied');
  // Dialog commented out
}
```

### After (Reactive with State)
```dart
try {
  final result = await _ocrService.pickImageAndRecognizeText(source);
  state = state.copyWith(
    imageFile: result.imageFile,
    recognizedText: result.recognizedText,
    isProcessing: false,
  );
} catch (e) {
  final errorMsg = e.toString();
  if (errorMsg.contains('PERMISSION_DENIED')) {
    state = state.copyWith(
      isPermissionDenied: true,
      isProcessing: false,
      error: 'Permission denied. Please enable access in settings.',
    );
  } else {
    state = state.copyWith(
      error: 'Failed to process image: $errorMsg',
      isProcessing: false,
    );
  }
}
```

**UI responds to state:**
```dart
if (state.error != null) {
  return ErrorBanner(
    message: state.error!,
    onDismiss: notifier.clearError,
  );
}
```

**Benefits:**
- ✅ Distinguishes permission errors
- ✅ Better error messages
- ✅ Error UI updates automatically
- ✅ User can dismiss errors
- ✅ No `mounted` check needed

---

## Text Recognition

### Before
```dart
final inputImage = InputImage.fromFilePath(pickedFile.path);
final recognizedText = await _textRecognizer.processImage(inputImage);

setState(() {
  _imageFile = pickedFile;
  _recognizedText = recognizedText.text;
});
```

### After (OcrService)
```dart
Future<String> recognizeText(XFile imageFile) async {
  try {
    final inputImage = InputImage.fromFilePath(imageFile.path);
    final recognizedText = await _textRecognizer.processImage(inputImage);
    
    debugPrint(
      '[OCR Recognition] Recognized ${recognizedText.text.length} characters',
    );
    
    return recognizedText.text;
  } catch (e) {
    debugPrint('[OCR Recognition] Error recognizing text: $e');
    rethrow;
  }
}
```

**Benefits:**
- ✅ Separate responsibility
- ✅ Better debugging with logs
- ✅ Reusable function
- ✅ Cleaner error handling

---

## UI Display

### Before
```dart
Expanded(
  child: _isProcessing
      ? const Center(
          child: CircularProgressIndicator.adaptive(),
        )
      : _imageFile == null
      ? const Center(child: Text('No image selected'))
      : SingleChildScrollView(
          child: Column(
            children: [
              Image.file(File(_imageFile!.path)),
              // ...
              ElevatedButton.icon(
                onPressed: _recognizedText.isEmpty ? null : () { 
                  // Copy logic
                },
              ),
            ],
          ),
        ),
),
```

### After (Cleaner Methods)
```dart
Expanded(
  child: _buildDisplayArea(ocrState),
),

const SizedBox(height: 16),
_buildActionButtons(context, ocrNotifier, ocrState),
```

With extracted methods:
```dart
Widget _buildDisplayArea(OcrState state) {
  if (state.isProcessing) {
    return const Center(child: CircularProgressIndicator.adaptive());
  }
  
  if (state.imageFile == null) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.image_not_supported_outlined, size: 64),
          const SizedBox(height: 16),
          Text('No image selected'),
        ],
      ),
    );
  }
  
  return SingleChildScrollView(
    child: Column(/* ... */),
  );
}
```

**Benefits:**
- ✅ Much more readable
- ✅ Easier to maintain
- ✅ Easier to test
- ✅ Better separation of concerns

---

## Summary Table

| Aspect | Before | After |
|--------|--------|-------|
| **Widget Type** | ConsumerStatefulWidget | ConsumerWidget |
| **State Management** | Manual setState | Riverpod Provider |
| **Permission Handling** | Mixed in UI | Centralized Service |
| **Error Handling** | SnackBar only | State + UI |
| **Code Organization** | Monolithic | Separated into services |
| **Testability** | Difficult | Easy |
| **Reusability** | Limited | High |
| **Error Messages** | Generic | Specific & helpful |
| **Resource Cleanup** | Manual dispose | Provider handles it |
| **Debug Logging** | Basic | Comprehensive |

---

## Migration Path for Other Screens

If you want to apply similar patterns to other screens:

1. **Extract business logic** → Create a `Service` class
2. **Create Riverpod Notifier** → Manage state reactively
3. **Refactor to ConsumerWidget** → Remove StatefulWidget
4. **Extract UI methods** → Keep build() clean
5. **Add error states** → Handle errors in state
6. **Test thoroughly** → Services are now testable

This pattern is now a template you can follow for other screens!
