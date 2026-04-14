import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:verifact_app/screens/verifier_result_screen.dart';
import 'package:verifact_app/services/ocr_service.dart';

/// Helper service that opens the device camera and can show the
/// captured image on a simple preview page.
class CameraService {
  CameraService._();

  static final ImagePicker _picker = ImagePicker();

  /// Launches the device camera and returns the picked file (or null).
  /// Keep this consistent with `ImagePickerService` (vanilla usage).
  static Future<XFile?> takePhoto({int imageQuality = 85}) async {
    try {
      final file = await _picker.pickImage(
        source: ImageSource.camera,
        imageQuality: imageQuality,
      );
      return file;
    } catch (_) {
      return null;
    }
  }

  /// Launches the camera and, if a photo is taken, navigates to
  /// a preview page that displays the captured image.
  static Future<void> openCameraAndShow(BuildContext context) async {
    final file = await takePhoto();
    if (file == null) return;

    await openPreview(context, file.path);
  }

  /// Opens the shared preview screen for any image path.
  static Future<void> openPreview(
    BuildContext context,
    String imagePath,
  ) async {
    await Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => _ImagePreviewScreen(imagePath: imagePath),
      ),
    );
  }
}

class _ImagePreviewScreen extends StatefulWidget {
  const _ImagePreviewScreen({required this.imagePath});

  final String imagePath;

  @override
  State<_ImagePreviewScreen> createState() => _ImagePreviewScreenState();
}

class _ImagePreviewScreenState extends State<_ImagePreviewScreen> {
  final OcrService _ocrService = OcrService();
  final TextEditingController _textController = TextEditingController();
  bool _isProcessing = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _runOcr();
  }

  @override
  void dispose() {
    _ocrService.dispose();
    _textController.dispose();
    super.dispose();
  }

  Future<void> _runOcr() async {
    try {
      final xfile = XFile(widget.imagePath);

      // Use the simpler full-text recognizer so we can display an editable
      // text area below the image (no per-block overlays).
      final recognized = await _ocrService.recognizeText(xfile);

      if (!mounted) return;
      setState(() {
        _textController.text = recognized;
        _isProcessing = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isProcessing = false;
        _error = e.toString();
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Photo Preview'),
      ),
      body: Column(
        children: [
          Expanded(
            child: Center(
              child: Image.file(
                File(widget.imagePath),
                fit: BoxFit.contain,
                width: double.infinity,
                height: double.infinity,
              ),
            ),
          ),

          if (_isProcessing) const LinearProgressIndicator(),
          if (_error != null)
            Padding(
              padding: const EdgeInsets.all(8),
              child: Text(
                _error!,
                style: const TextStyle(color: Colors.redAccent),
                textAlign: TextAlign.center,
              ),
            ),

          Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                TextFormField(
                  controller: _textController,
                  maxLines: 6,
                  minLines: 3,
                  decoration: const InputDecoration(
                    border: OutlineInputBorder(),
                    labelText: 'Extracted text',
                  ),
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    Expanded(
                      child: ElevatedButton(
                        onPressed: () {
                          final claim = _textController.text.trim();
                          if (claim.isEmpty) return;
                          Navigator.of(context).push(
                            MaterialPageRoute<void>(
                              builder: (_) => VerifierResultScreen(claim: claim),
                            ),
                          );
                        },
                        child: const Text('Go'),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
