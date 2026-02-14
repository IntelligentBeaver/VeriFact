import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart'
    as ml_kit;
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:verifact_app/providers/ocr_provider.dart';

class OcrPickerResult {
  OcrPickerResult({
    required this.imageFile,
    required this.recognizedText,
    required this.textBlocks,
    required this.imageWidth,
    required this.imageHeight,
  });
  final XFile imageFile;
  final String recognizedText;
  final List<TextBlock> textBlocks;
  final double imageWidth;
  final double imageHeight;
}

class OcrService {

  OcrService() {
    _initializeTextRecognizer();
  }
  final ImagePicker _picker = ImagePicker();
  late ml_kit.TextRecognizer _textRecognizer;
  bool _isInitialized = false;

  void _initializeTextRecognizer() {
    if (!_isInitialized) {
      _textRecognizer = ml_kit.TextRecognizer();
      _isInitialized = true;
    }
  }

  /// Request camera or gallery permission with detailed handling
  Future<bool> requestPermission(ImageSource source) async {
    if (source == ImageSource.camera) {
      return _requestCameraPermission();
    } else {
      return _requestGalleryPermission();
    }
  }

  /// Request camera permission with proper iOS/Android handling
  Future<bool> _requestCameraPermission() async {
    final status = await Permission.camera.request();

    debugPrint('[OCR Permission] Camera permission status: $status');

    if (status.isDenied) {
      return false;
    } else if (status.isPermanentlyDenied) {
      throw Exception('PERMISSION_DENIED');
    }

    return status.isGranted;
  }

  /// Request gallery/photos permission with platform-specific logic
  Future<bool> _requestGalleryPermission() async {
    if (Platform.isIOS) {
      return _requestGalleryPermissionIOS();
    } else {
      return _requestGalleryPermissionAndroid();
    }
  }

  Future<bool> _requestGalleryPermissionIOS() async {
    final photosStatus = await Permission.photos.request();

    debugPrint('[OCR Permission] iOS photos permission status: $photosStatus');

    // iOS 14+ has 'limited' status; treat as allowed
    if (photosStatus == PermissionStatus.limited) return true;

    if (photosStatus.isDenied) {
      return false;
    }

    if (photosStatus.isPermanentlyDenied) {
      // iOS PHPicker might still work even with denied permission
      return true;
    }

    return photosStatus.isGranted;
  }

  Future<bool> _requestGalleryPermissionAndroid() async {
    final imageStatus = await Permission.photos.status;
    final extStorageStatus = await Permission.manageExternalStorage.status;
    final storageStatus = await Permission.storage.status;

    debugPrint(
      '[OCR Permission] Android - photos: $imageStatus, extStorage: $extStorageStatus, storage: $storageStatus',
    );

    // Request all storage-related permissions
    final requestedImage = await Permission.photos.request();
    final requestedExtStorage = await Permission.manageExternalStorage
        .request();
    final requestedStorage = await Permission.storage.request();

    // Check if any permission is permanently denied
    if (requestedImage.isPermanentlyDenied ||
        requestedExtStorage.isPermanentlyDenied ||
        requestedStorage.isPermanentlyDenied) {
      throw Exception('PERMISSION_DENIED');
    }

    // Return true if at least one permission is granted
    return requestedImage.isGranted ||
        requestedExtStorage.isGranted ||
        requestedStorage.isGranted;
  }

  /// Check if permission is already granted
  Future<bool> hasPermission(ImageSource source) async {
    if (source == ImageSource.camera) {
      final status = await Permission.camera.status;
      return status.isGranted;
    } else {
      if (Platform.isIOS) {
        final status = await Permission.photos.status;
        return status.isGranted || status.isLimited;
      } else {
        final photosStatus = await Permission.photos.status;
        final extStorageStatus = await Permission.manageExternalStorage.status;
        final storageStatus = await Permission.storage.status;

        return photosStatus.isGranted ||
            extStorageStatus.isGranted ||
            storageStatus.isGranted;
      }
    }
  }

  /// Pick image with robust error handling and compression
  Future<XFile?> pickImage(ImageSource source) async {
    try {
      // Skip permission check for iOS gallery on PHPicker
      if (!(Platform.isIOS && source == ImageSource.gallery)) {
        final hasPermissionResult = await hasPermission(source);
        if (!hasPermissionResult) {
          final permissionGranted = await requestPermission(source);
          if (!permissionGranted) {
            throw Exception('PERMISSION_DENIED');
          }
        }
      }

      final pickedFile = await _picker
          .pickImage(
            source: source,
            maxWidth: 2048,
            maxHeight: 2048,
            imageQuality: 85,
          )
          .timeout(
            const Duration(seconds: 30),
            onTimeout: () {
              debugPrint('[OCR Picker] pickImage timed out after 30s');
              return null;
            },
          );

      if (pickedFile == null) {
        debugPrint('[OCR Picker] User cancelled image selection');
        return null;
      }

      debugPrint('[OCR Picker] Image picked: ${pickedFile.path}');

      // Verify file exists and is accessible
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

  /// Recognize text from picked image and extract bounding boxes
  Future<List<TextBlock>> recognizeTextWithBlocks(XFile imageFile) async {
    try {
      final inputImage = ml_kit.InputImage.fromFilePath(imageFile.path);
      final recognizedText = await _textRecognizer.processImage(inputImage);

      // Get image dimensions from the decoded image
      // For now, we'll read the file size info from the raw image file
      final imageData = File(imageFile.path).readAsBytesSync();
      final decodedImage = _decodeImageSize(imageData);

      final imageWidth = decodedImage?['width']?.toDouble() ?? 1.0;
      final imageHeight = decodedImage?['height']?.toDouble() ?? 1.0;

      debugPrint(
        '[OCR Recognition] Image dimensions: $imageWidth x $imageHeight',
      );

      final textBlocks = <TextBlock>[];

      for (final block in recognizedText.blocks) {
        final boundingBox = block.boundingBox;

        textBlocks.add(
          TextBlock(
            text: block.text,
            boundingBox: boundingBox,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
          ),
        );
      }

      debugPrint(
        '[OCR Recognition] Recognized ${recognizedText.text.length} characters in ${textBlocks.length} blocks',
      );

      return textBlocks;
    } catch (e) {
      debugPrint('[OCR Recognition] Error recognizing text: $e');
      rethrow;
    }
  }

  /// Decode image size from raw bytes (supports JPEG and PNG)
  Map<String, int>? _decodeImageSize(List<int> bytes) {
    if (bytes.isEmpty) return null;

    // Check for JPEG magic bytes
    if (bytes[0] == 0xFF && bytes[1] == 0xD8) {
      return _getJpegDimensions(bytes);
    }

    // Check for PNG magic bytes
    if (bytes[0] == 0x89 &&
        bytes[1] == 0x50 &&
        bytes[2] == 0x4E &&
        bytes[3] == 0x47) {
      return _getPngDimensions(bytes);
    }

    return null;
  }

  /// Extract dimensions from JPEG
  Map<String, int>? _getJpegDimensions(List<int> bytes) {
    var i = 2;
    while (i < bytes.length) {
      if (bytes[i] == 0xFF) {
        if (bytes[i + 1] == 0xC0 || bytes[i + 1] == 0xC2) {
          // Found SOF (Start of Frame)
          final height = (bytes[i + 5] << 8) | bytes[i + 6];
          final width = (bytes[i + 7] << 8) | bytes[i + 8];
          return {'width': width, 'height': height};
        }
        i += 2;
      } else {
        i++;
      }
    }
    return null;
  }

  /// Extract dimensions from PNG
  Map<String, int>? _getPngDimensions(List<int> bytes) {
    if (bytes.length < 24) return null;
    // PNG IHDR chunk is always at position 16-24
    final width =
        (bytes[16] << 24) | (bytes[17] << 16) | (bytes[18] << 8) | bytes[19];
    final height =
        (bytes[20] << 24) | (bytes[21] << 16) | (bytes[22] << 8) | bytes[23];
    return {'width': width, 'height': height};
  }

  /// Recognize text from picked image (legacy method - returns full text)
  Future<String> recognizeText(XFile imageFile) async {
    try {
      final inputImage = ml_kit.InputImage.fromFilePath(imageFile.path);
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

  /// Combined method: pick image and recognize text with blocks
  Future<OcrPickerResult> pickImageAndRecognizeText(ImageSource source) async {
    try {
      final imageFile = await pickImage(source);

      if (imageFile == null) {
        throw Exception('No image selected');
      }

      final recognizedText = await recognizeText(imageFile);
      final textBlocks = await recognizeTextWithBlocks(imageFile);

      // Get image dimensions for the result
      final imageData = File(imageFile.path).readAsBytesSync();
      final decodedImage = _decodeImageSize(imageData);
      final imageWidth = decodedImage?['width']?.toDouble() ?? 1.0;
      final imageHeight = decodedImage?['height']?.toDouble() ?? 1.0;

      return OcrPickerResult(
        imageFile: imageFile,
        recognizedText: recognizedText,
        textBlocks: textBlocks,
        imageWidth: imageWidth,
        imageHeight: imageHeight,
      );
    } catch (e) {
      debugPrint('[OCR Service] Error in pickImageAndRecognizeText: $e');
      rethrow;
    }
  }

  /// Clean up resources
  void dispose() {
    if (_isInitialized) {
      _textRecognizer.close();
      _isInitialized = false;
    }
  }
}
