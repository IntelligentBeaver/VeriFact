import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart';
import 'package:verifact_app/services/ocr_service.dart';

/// Text block with bounding box information
class TextBlock {
  const TextBlock({
    required this.text,
    required this.boundingBox,
    required this.imageWidth,
    required this.imageHeight,
  });

  final String text;
  final Rect boundingBox; // Absolute pixel coordinates from MLkit
  final double imageWidth;
  final double imageHeight;

  /// Get normalized coordinates (0-1) for rendering
  Rect get normalizedBoundingBox {
    return Rect.fromLTRB(
      boundingBox.left / imageWidth,
      boundingBox.top / imageHeight,
      boundingBox.right / imageWidth,
      boundingBox.bottom / imageHeight,
    );
  }

  @override
  String toString() =>
      'TextBlock(text: $text, box: $boundingBox, size: ${imageWidth}x$imageHeight)';
}

/// State model for OCR screen
class OcrState {
  const OcrState({
    this.imageFile,
    this.recognizedText = '',
    this.textBlocks = const [],
    this.isProcessing = false,
    this.error,
    this.isPermissionDenied = false,
    this.selectedBlockIndex,
    this.selectedTextStart,
    this.selectedTextEnd,
  });
  final XFile? imageFile;
  final String recognizedText;
  final List<TextBlock> textBlocks;
  final bool isProcessing;
  final String? error;
  final bool isPermissionDenied;
  final int? selectedBlockIndex;
  final int? selectedTextStart;
  final int? selectedTextEnd;

  OcrState copyWith({
    XFile? imageFile,
    String? recognizedText,
    List<TextBlock>? textBlocks,
    bool? isProcessing,
    String? error,
    bool? isPermissionDenied,
    int? selectedBlockIndex,
    int? selectedTextStart,
    int? selectedTextEnd,
  }) {
    return OcrState(
      imageFile: imageFile ?? this.imageFile,
      recognizedText: recognizedText ?? this.recognizedText,
      textBlocks: textBlocks ?? this.textBlocks,
      isProcessing: isProcessing ?? this.isProcessing,
      error: error,
      isPermissionDenied: isPermissionDenied ?? this.isPermissionDenied,
      selectedBlockIndex: selectedBlockIndex ?? this.selectedBlockIndex,
      selectedTextStart: selectedTextStart ?? this.selectedTextStart,
      selectedTextEnd: selectedTextEnd ?? this.selectedTextEnd,
    );
  }

  @override
  String toString() =>
      'OcrState(imageFile: ${imageFile?.name}, recognizedText: ${recognizedText.length} chars, textBlocks: ${textBlocks.length}, isProcessing: $isProcessing, error: $error, isPermissionDenied: $isPermissionDenied)';
}

/// OCR Service Provider - Singleton instance
final ocrServiceProvider = Provider<OcrService>((ref) {
  return OcrService();
});

/// OCR Notifier class for managing state
class OcrNotifier extends Notifier<OcrState> {
  late OcrService _ocrService;

  @override
  OcrState build() {
    _ocrService = ref.watch(ocrServiceProvider);
    return const OcrState();
  }

  /// Pick image from gallery or camera and recognize text
  Future<void> pickAndRecognizeImage(ImageSource source) async {
    state = state.copyWith(isProcessing: true, error: null);

    try {
      final result = await _ocrService.pickImageAndRecognizeText(source);

      state = state.copyWith(
        imageFile: result.imageFile,
        recognizedText: result.recognizedText,
        textBlocks: result.textBlocks,
        isProcessing: false,
        isPermissionDenied: false,
      );
    } catch (e) {
      if (e.toString().contains('PERMISSION_DENIED')) {
        state = state.copyWith(
          isPermissionDenied: true,
          isProcessing: false,
          error: 'Permission denied. Please enable access in settings.',
        );
      } else {
        state = state.copyWith(
          error: 'Failed to process image: $e',

          isProcessing: false,
        );
      }
    }
  }

  /// Reset the OCR state
  void reset() {
    state = const OcrState();
  }

  /// Clear only the error message
  void clearError() {
    state = state.copyWith(error: null);
  }

  /// Select a text block by index
  void selectTextBlock(int? index) {
    state = state.copyWith(
      selectedBlockIndex: index,
      selectedTextStart: null,
      selectedTextEnd: null,
    );
  }

  /// Select text range within the selected block
  void selectTextRange(int? start, int? end) {
    if (start != null && end != null) {
      // Ensure start <= end
      final s = start < end ? start : end;
      final e = start < end ? end : start;
      state = state.copyWith(selectedTextStart: s, selectedTextEnd: e);
    } else {
      state = state.copyWith(selectedTextStart: null, selectedTextEnd: null);
    }
  }

  /// Get the selected text from the selected block
  String getSelectedText() {
    final blockIndex = state.selectedBlockIndex;
    final start = state.selectedTextStart;
    final end = state.selectedTextEnd;

    if (blockIndex != null &&
        blockIndex >= 0 &&
        blockIndex < state.textBlocks.length &&
        start != null &&
        end != null) {
      final block = state.textBlocks[blockIndex];
      if (start >= 0 && end <= block.text.length && start < end) {
        return block.text.substring(start, end);
      }
    }
    return '';
  }

  /// Get the selected text block, if any
  TextBlock? getSelectedTextBlock() {
    final selectedIndex = state.selectedBlockIndex;
    if (selectedIndex != null &&
        selectedIndex >= 0 &&
        selectedIndex < state.textBlocks.length) {
      return state.textBlocks[selectedIndex];
    }
    return null;
  }

  /// Retake image (reset but keep processing state if needed)
  void retakeImage() {
    state = state.copyWith(
      imageFile: null,
      recognizedText: '',
      textBlocks: const [],
      selectedBlockIndex: null,
      selectedTextStart: null,
      selectedTextEnd: null,
      error: null,
    );
  }

  /// Dispose resources
  void dispose() {
    _ocrService.dispose();
  }
}

/// OCR Notifier Provider - Using Notifier for Riverpod v3
final ocrNotifierProvider = NotifierProvider<OcrNotifier, OcrState>(() {
  return OcrNotifier();
});

// Derived providers for convenient access
final isOcrProcessingProvider = Provider<bool>((ref) {
  return ref.watch(ocrNotifierProvider).isProcessing;
});

final hasRecognizedTextProvider = Provider<bool>((ref) {
  return ref.watch(ocrNotifierProvider).recognizedText.isNotEmpty;
});

final recognizedTextProvider = Provider<String>((ref) {
  return ref.watch(ocrNotifierProvider).recognizedText;
});

final ocrErrorProvider = Provider<String?>((ref) {
  return ref.watch(ocrNotifierProvider).error;
});
