import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart';
import 'package:verifact_app/providers/ocr_provider.dart';
import 'package:verifact_app/utils/helpers/helper_functions.dart';

/// Custom painter for drawing text box overlays on the image
class TextBoxOverlayPainter extends CustomPainter {
  TextBoxOverlayPainter({
    required this.textBlocks,
    required this.selectedIndex,
  });

  final List<TextBlock> textBlocks;
  final int? selectedIndex;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.blue.withAlpha(100)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    final selectedPaint = Paint()
      ..color = Colors.red.withAlpha(200)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    final fillPaint = Paint()
      ..color = Colors.green.withAlpha(40)
      ..style = PaintingStyle.fill;

    // Draw all text boxes
    for (int i = 0; i < textBlocks.length; i++) {
      final block = textBlocks[i];
      final isSelected = i == selectedIndex;

      // Use normalized coordinates for rendering
      final normalizedBox = block.normalizedBoundingBox;
      final rect = Rect.fromLTRB(
        normalizedBox.left * size.width,
        normalizedBox.top * size.height,
        normalizedBox.right * size.width,
        normalizedBox.bottom * size.height,
      );

      // Draw fill for selected box
      if (isSelected) {
        canvas.drawRect(rect, fillPaint);
      }

      // Draw border
      final borderPaint = isSelected ? selectedPaint : paint;
      canvas.drawRect(rect, borderPaint);

      // Draw text label for selected box
      if (isSelected) {
        _drawTextLabel(canvas, rect, block.text);
      }
    }
  }

  void _drawTextLabel(Canvas canvas, Rect rect, String text) {
    final displayText = text.length > 30 ? '${text.substring(0, 30)}...' : text;

    final textPainter = TextPainter(
      text: TextSpan(
        text: displayText,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 11,
          fontWeight: FontWeight.w500,
          backgroundColor: Colors.black87,
          height: 1.2,
        ),
      ),
      textDirection: TextDirection.ltr,
      maxLines: 1,
    );

    textPainter.layout(maxWidth: rect.width);

    // Position label above the box, with some padding
    final labelY = (rect.top - textPainter.height - 6).clamp(
      0.0,
      double.infinity,
    );
    final labelX = rect.left.clamp(0.0, double.infinity);

    // Draw background for label
    final labelPaint = Paint()
      ..color = Colors.black87
      ..style = PaintingStyle.fill;

    canvas.drawRect(
      Rect.fromLTWH(
        labelX,
        labelY,
        textPainter.width + 8,
        textPainter.height + 4,
      ),
      labelPaint,
    );

    textPainter.paint(canvas, Offset(labelX + 4, labelY + 2));
  }

  @override
  bool shouldRepaint(TextBoxOverlayPainter oldDelegate) {
    return textBlocks != oldDelegate.textBlocks ||
        selectedIndex != oldDelegate.selectedIndex;
  }
}

class OcrScreen extends ConsumerWidget {
  const OcrScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final ocrState = ref.watch(ocrNotifierProvider);
    final ocrNotifier = ref.read(ocrNotifierProvider.notifier);

    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              /// `Recognized Display Area`
              Expanded(
                child: _buildDisplayArea(ocrState, ocrNotifier),
              ),

              const SizedBox(height: 16),

              /// `Action Buttons`
              _buildActionButtons(context, ocrNotifier, ocrState),
            ],
          ),
        ),
      ),
    );
  }

  /// Build the display area showing image and recognized text with overlay
  Widget _buildDisplayArea(OcrState state, OcrNotifier notifier) {
    if (state.isProcessing) {
      return const Center(
        child: CircularProgressIndicator.adaptive(),
      );
    }

    if (state.imageFile == null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.image_not_supported_outlined,
              size: 64,
              color: Colors.grey[400],
            ),
            const SizedBox(height: 16),
            Text(
              'No image selected',
              style: TextStyle(color: Colors.grey[600]),
            ),
          ],
        ),
      );
    }

    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          /// `Image Preview with Text Box Overlay`
          ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: _buildImageWithOverlay(state, notifier),
          ),
          const SizedBox(height: 12),

          /// `Text Blocks List`
          if (state.textBlocks.isNotEmpty)
            _buildTextBlocksList(state, notifier),

          /// `Recognized Text Container`
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.black87,
              borderRadius: BorderRadius.circular(8),
            ),
            child: SingleChildScrollView(
              child: Text(
                state.recognizedText.isEmpty
                    ? 'No text recognized.'
                    : state.recognizedText,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 14,
                  height: 1.5,
                ),
              ),
            ),
          ),
          const SizedBox(height: 12),

          /// `Copy and Retry Buttons`
          Wrap(
            crossAxisAlignment: WrapCrossAlignment.center,
            spacing: 8,
            runSpacing: 8,
            children: [
              ElevatedButton.icon(
                onPressed: state.recognizedText.isEmpty
                    ? null
                    : () {
                        Clipboard.setData(
                          ClipboardData(text: state.recognizedText),
                        );
                        showInfoSnackbar('Text copied to clipboard');
                      },
                icon: const Icon(Icons.copy),
                label: const Text('Copy'),
              ),
              FilledButton.tonal(
                onPressed: () {
                  notifier.retakeImage();
                },
                child: const Text('Retake'),
              ),
            ],
          ),
          const SizedBox(height: 52),
        ],
      ),
    );
  }

  /// Build the image with overlay on top
  Widget _buildImageWithOverlay(OcrState state, OcrNotifier notifier) {
    return LayoutBuilder(
      builder: (context, constraints) {
        // Get actual image dimensions from the first text block
        final firstBlock = state.textBlocks.isNotEmpty
            ? state.textBlocks[0]
            : null;
        final actualImageWidth = firstBlock?.imageWidth ?? 1000.0;
        final actualImageHeight = firstBlock?.imageHeight ?? 1000.0;

        // Calculate display height based on actual image aspect ratio
        final displayWidth = constraints.maxWidth;
        final aspectRatio = actualImageHeight / actualImageWidth;
        final displayHeight = displayWidth * aspectRatio;

        return SizedBox(
          width: displayWidth,
          height: displayHeight,
          child: Stack(
            fit: StackFit.expand,
            children: [
              Image.file(
                File(state.imageFile!.path),
                fit: BoxFit.cover,
                errorBuilder: (context, error, stackTrace) {
                  return Container(
                    color: Colors.grey[300],
                    child: const Center(child: Text('Failed to load image')),
                  );
                },
              ),
              // Text box overlay with character-level mapping (Google Lens style)
              if (state.textBlocks.isNotEmpty)
                Positioned.fill(
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      // Overlay painter for visual feedback
                      GestureDetector(
                        onTapDown: (details) {
                          _selectTextBoxAtPositionWithIndex(
                            details.localPosition,
                            state,
                            notifier,
                            displayWidth,
                            displayHeight,
                          );
                        },
                        child: CustomPaint(
                          painter: TextBoxOverlayPainter(
                            textBlocks: state.textBlocks,
                            selectedIndex: state.selectedBlockIndex,
                          ),
                        ),
                      ),
                      // Character-level selectable text overlays
                      ...state.textBlocks.asMap().entries.map(
                        (entry) {
                          final index = entry.key;
                          final block = entry.value;
                          return _buildCharacterLevelOverlay(
                            block,
                            index,
                            displayWidth,
                            displayHeight,
                            state.selectedBlockIndex == index,
                          );
                        },
                      ).toList(),
                    ],
                  ),
                ),
            ],
          ),
        );
      },
    );
  }

  /// Build the list of text blocks for easy selection
  Widget _buildTextBlocksList(OcrState state, OcrNotifier notifier) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Colors.grey[100],
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.grey[300]!),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Detected Text Regions (${state.textBlocks.length})',
                style: const TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.bold,
                  color: Colors.black87,
                ),
              ),
              if (state.selectedBlockIndex != null)
                GestureDetector(
                  onTap: () => notifier.selectTextBlock(null),
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 8,
                      vertical: 4,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.red[100],
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Text(
                      'Clear',
                      style: TextStyle(
                        fontSize: 11,
                        color: Colors.red[700],
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 10),
          SizedBox(
            height: 90,
            child: ListView.builder(
              scrollDirection: Axis.horizontal,
              itemCount: state.textBlocks.length,
              itemBuilder: (context, index) {
                final block = state.textBlocks[index];
                final isSelected = index == state.selectedBlockIndex;

                return GestureDetector(
                  onTap: () =>
                      notifier.selectTextBlock(isSelected ? null : index),
                  child: Container(
                    margin: const EdgeInsets.symmetric(horizontal: 5),
                    padding: const EdgeInsets.all(8),
                    constraints: const BoxConstraints(maxWidth: 140),
                    decoration: BoxDecoration(
                      color: isSelected ? Colors.blue : Colors.white,
                      border: Border.all(
                        color: isSelected
                            ? Colors.blue[600]!
                            : Colors.grey[400]!,
                        width: isSelected ? 2 : 1,
                      ),
                      borderRadius: BorderRadius.circular(8),
                      boxShadow: isSelected
                          ? [
                              BoxShadow(
                                color: Colors.blue.withAlpha(100),
                                blurRadius: 4,
                                spreadRadius: 1,
                              ),
                            ]
                          : [],
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Box ${index + 1}',
                          style: TextStyle(
                            fontSize: 10,
                            fontWeight: FontWeight.bold,
                            color: isSelected ? Colors.white : Colors.grey[600],
                          ),
                        ),
                        const SizedBox(height: 4),
                        Expanded(
                          child: SingleChildScrollView(
                            child: Text(
                              block.text,
                              style: TextStyle(
                                fontSize: 11,
                                color: isSelected
                                    ? Colors.white
                                    : Colors.black87,
                                height: 1.3,
                              ),
                              maxLines: 3,
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  /// Build character-level selectable text overlay (Google Lens style)
  Widget _buildCharacterLevelOverlay(
    TextBlock block,
    int blockIndex,
    double containerWidth,
    double containerHeight,
    bool isSelected,
  ) {
    final normalizedBox = block.normalizedBoundingBox;
    final boxWidth =
        (normalizedBox.right - normalizedBox.left) * containerWidth;
    final boxHeight =
        (normalizedBox.bottom - normalizedBox.top) * containerHeight;
    final boxLeft = normalizedBox.left * containerWidth;
    final boxTop = normalizedBox.top * containerHeight;

    // Count lines in the text to scale font appropriately
    final lines = block.text.split('\n').length;

    // Calculate font size based on box dimensions and number of lines
    // For single line: use full height
    // For multiple lines: reduce proportionally
    final lineHeight = 1.2; // Line spacing multiplier
    final totalLineHeight = lines * lineHeight;
    final fontSize = (boxHeight / totalLineHeight) * 1;

    return Positioned(
      left: boxLeft,
      top: boxTop,
      width: boxWidth,
      height: boxHeight,
      child: GestureDetector(
        onTap: () {
          // Tapping selects the text block
        },
        child: Container(
          // Light overlay for selected block
          color: isSelected ? Colors.blue.withAlpha(25) : Colors.transparent,
          alignment: Alignment.center,
          padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
          child: SelectableText(
            block.text,
            style: TextStyle(
              fontSize: fontSize,
              color: isSelected ? Colors.blue : Colors.transparent,
              fontWeight: FontWeight.w600,
              height: lineHeight,
              leadingDistribution: TextLeadingDistribution.even,
            ),
            textAlign: TextAlign.center,
            strutStyle: StrutStyle(
              fontSize: fontSize,
              height: lineHeight,
              leading: 0,
            ),
          ),
        ),
      ),
    );
  }

  /// Select text box at a given position on the image and return selected index
  int? _selectTextBoxAtPositionWithIndex(
    Offset position,
    OcrState state,
    OcrNotifier notifier,
    double containerWidth,
    double containerHeight,
  ) {
    if (state.textBlocks.isEmpty) return null;

    // Check which text block was tapped by converting position to normalized coordinates
    for (int i = 0; i < state.textBlocks.length; i++) {
      final block = state.textBlocks[i];

      // Get normalized bounding box
      final normalizedBox = block.normalizedBoundingBox;

      // Convert to container coordinates
      final rect = Rect.fromLTRB(
        normalizedBox.left * containerWidth,
        normalizedBox.top * containerHeight,
        normalizedBox.right * containerWidth,
        normalizedBox.bottom * containerHeight,
      );

      // Check if tap is within this box
      if (rect.contains(position)) {
        notifier.selectTextBlock(i);
        return i;
      }
    }

    // If no block was tapped, deselect
    notifier.selectTextBlock(null);
    return null;
  }

  /// Build action buttons
  Widget _buildActionButtons(
    BuildContext context,
    OcrNotifier notifier,
    OcrState state,
  ) {
    // Show error banner if exists
    if (state.error != null) {
      return Column(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.red[50],
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: Colors.red[200]!),
            ),
            child: Row(
              children: [
                Icon(Icons.error_outline, color: Colors.red[700]),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    state.error!,
                    style: TextStyle(color: Colors.red[700]),
                  ),
                ),
                IconButton(
                  iconSize: 20,
                  icon: const Icon(Icons.close),
                  onPressed: notifier.clearError,
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          _buildPickerButtons(notifier, state),
        ],
      );
    }

    return _buildPickerButtons(notifier, state);
  }

  /// Build the image picker buttons
  Widget _buildPickerButtons(OcrNotifier notifier, OcrState state) {
    final isLoading = state.isProcessing;

    return Align(
      alignment: Alignment.bottomCenter,
      child: Wrap(
        alignment: WrapAlignment.center,
        spacing: 12,
        runSpacing: 12,
        children: [
          /// `Gallery Button`
          ElevatedButton.icon(
            onPressed: isLoading
                ? null
                : () => notifier.pickAndRecognizeImage(ImageSource.gallery),
            icon: const Icon(Icons.photo_library),
            label: const Text('Pick Image'),
          ),

          /// `Camera Button`
          ElevatedButton.icon(
            onPressed: isLoading
                ? null
                : () => notifier.pickAndRecognizeImage(ImageSource.camera),
            icon: const Icon(Icons.camera_alt),
            label: const Text('Use Camera'),
          ),
        ],
      ),
    );
  }
}
