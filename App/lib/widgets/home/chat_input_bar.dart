import 'package:flutter/material.dart';
import 'package:lucide_icons_flutter/lucide_icons.dart';
import 'package:speech_to_text/speech_recognition_result.dart' as stt;
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:verifact_app/extensions/context_extensions.dart';
import 'package:verifact_app/utils/constants/sizes.dart';

class ChatInputBar extends StatefulWidget {
  const ChatInputBar({
    required this.controller,
    required this.hintText,
    required this.onPlusTap,
    required this.onSubmitted,
    super.key,
  });

  final TextEditingController controller;
  final String hintText;
  final VoidCallback onPlusTap;
  final ValueChanged<String> onSubmitted;

  @override
  State<ChatInputBar> createState() => _ChatInputBarState();
}

class _ChatInputBarState extends State<ChatInputBar> {
  late stt.SpeechToText _speech;
  bool _listening = false;

  @override
  void initState() {
    super.initState();
    _speech = stt.SpeechToText();
    _initSpeech();
  }

  Future<void> _initSpeech() async {
    try {
      await _speech.initialize();
    } catch (_) {}
  }

  void _toggleListening() async {
    if (!_listening) {
      final available = await _speech.initialize();
      if (!available) return;
      setState(() => _listening = true);
      _speech.listen(
        onResult: (stt.SpeechRecognitionResult result) {
          final text = result.recognizedWords;
          widget.controller.text = text;
          widget.controller.selection = TextSelection.fromPosition(
            TextPosition(offset: text.length),
          );
        },
        localeId: 'en_US',
      );
    } else {
      await _speech.stop();
      setState(() => _listening = false);
      widget.onSubmitted(widget.controller.text.trim());
    }
  }

  @override
  void dispose() {
    _speech.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.symmetric(
        horizontal: AppSizes.md,
        vertical: AppSizes.smMd,
      ),
      decoration: BoxDecoration(
        color: context.color.surface,
        borderRadius: BorderRadius.circular(AppSizes.borderRadiusLg),
        border: Border.all(color: context.color.outlineVariant),
      ),
      child: Row(
        children: [
          Icon(
            LucideIcons.search,
            size: AppSizes.iconSm,
            color: context.color.onSurfaceVariant,
          ),
          SizedBox(width: AppSizes.smMd),
          Expanded(
            child: TextField(
              controller: widget.controller,
              onSubmitted: widget.onSubmitted,
              textInputAction: TextInputAction.search,
              style: context.text.bodyMedium?.copyWith(
                color: context.color.onSurface,
              ),
              decoration: InputDecoration(
                hintText: widget.hintText,
                hintStyle: context.text.bodySmall?.copyWith(
                  color: context.color.onSurfaceVariant,
                ),
                isDense: true,
                border: InputBorder.none,
              ),
            ),
          ),
          SizedBox(width: AppSizes.smMd),
          InkWell(
            onTap: widget.onPlusTap,
            borderRadius: BorderRadius.circular(AppSizes.borderRadiusMd),
            child: Container(
              width: AppSizes.iconLg,
              height: AppSizes.iconLg,
              decoration: BoxDecoration(
                color: context.color.surfaceVariant,
                borderRadius: BorderRadius.circular(AppSizes.borderRadiusMd),
              ),
              child: Icon(
                LucideIcons.plus,
                size: AppSizes.iconSm,
                color: context.color.onSurfaceVariant,
              ),
            ),
          ),
          const SizedBox(width: 8),
          GestureDetector(
            onTap: _toggleListening,
            child: CircleAvatar(
              radius: 18,
              backgroundColor: _listening
                  ? context.color.primary
                  : context.color.surfaceVariant,
              child: Icon(
                LucideIcons.mic,
                size: 18,
                color: _listening
                    ? context.color.onPrimary
                    : context.color.onSurfaceVariant,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
