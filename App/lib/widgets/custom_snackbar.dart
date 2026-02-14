import 'dart:async';

import 'package:flutter/material.dart';
import 'package:lucide_icons_flutter/lucide_icons.dart';
import 'package:verifact_app/extensions/context_extensions.dart';
import 'package:verifact_app/utils/constants/app_globals.dart';
import 'package:verifact_app/utils/constants/colors.dart';
import 'package:verifact_app/utils/constants/sizes.dart';

class TopSnackbar {
  static OverlayEntry? _entry;

  /// Show snackbar. Prefer passing [navKey] if you have a global navigator key.
  static void show(
    BuildContext context,
    Widget child, {
    Duration duration = const Duration(milliseconds: 2000),
  }) {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      // Prefer explicit navKey overlay (most reliable)
      final overlay =
          navigatorKey.currentState?.overlay ??
          Overlay.of(context, rootOverlay: true);

      // remove existing then insert new
      _remove();

      _entry = OverlayEntry(
        builder: (_) => _TopSnackbarWidget(
          duration: duration,
          onRequestClose: _remove,
          child: SafeArea(
            bottom: false,
            child: Padding(
              padding: EdgeInsets.symmetric(
                horizontal: AppSizes.smMd,
                vertical: AppSizes.md,
              ),
              child: child,
            ),
          ),
        ),
      );

      overlay.insert(_entry!);
    });
  }

  static void _remove() {
    _entry?.remove();
    _entry = null;
  }

  // convenience builders
  static Widget error(String message) => _defaultContainer(
    message,
    color: navigatorKey.currentContext!.color.onErrorContainer,
    containerColor: navigatorKey.currentContext!.color.errorContainer,
    icon: LucideIcons.circleX400,
  );

  static Widget success(String message) => _defaultContainer(
    message,
    color: AppColors.onSuccess,
    containerColor: AppColors.success,
    icon: LucideIcons.circleCheck400,
  );

  static Widget info(String message) => _defaultContainer(
    message,
    color: AppColors.onPrimary,
    containerColor: AppColors.info,
    icon: LucideIcons.circleAlert400,
  );

  static Widget _defaultContainer(
    String message, {
    required Color color,
    required Color containerColor,
    required IconData icon,
  }) {
    return Material(
      elevation: 1,
      borderRadius: BorderRadius.circular(AppSizes.borderRadiusMd),
      child: Container(
        decoration: BoxDecoration(
          color: containerColor,
          borderRadius: BorderRadius.circular(AppSizes.borderRadiusMd),
        ),
        padding: EdgeInsets.symmetric(
          vertical: AppSizes.smMd,
          horizontal: AppSizes.md,
        ),
        child: Row(
          children: [
            Icon(icon, color: color),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                message,
                style: navigatorKey.currentContext!.text.bodyMedium?.copyWith(
                  color: color,
                ),
              ),
            ),
            GestureDetector(
              onTap: _remove,
              child: Icon(Icons.close, color: color),
            ),
          ],
        ),
      ),
    );
  }
}

class _TopSnackbarWidget extends StatefulWidget {
  const _TopSnackbarWidget({
    required this.child,
    required this.onRequestClose,
    required this.duration,
  });
  final Widget child;
  final VoidCallback onRequestClose;
  final Duration duration;

  @override
  State<_TopSnackbarWidget> createState() => _TopSnackbarWidgetState();
}

class _TopSnackbarWidgetState extends State<_TopSnackbarWidget>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;
  late final Animation<Offset> _offset;
  late final Animation<double> _fade;

  @override
  void initState() {
    super.initState();

    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 300),
    );
    _offset = Tween<Offset>(begin: const Offset(0, -0.4), end: Offset.zero)
        .animate(
          CurvedAnimation(
            parent: _ctrl,
            curve: Curves.easeOutQuart,
          ),
        );
    _fade = CurvedAnimation(
      parent: _ctrl,
      curve: Curves.easeOutQuart,
    );

    // start show animation
    unawaited(_ctrl.forward());

    // schedule auto-close: wait [duration], then reverse animation, then request close
    Future.delayed(widget.duration, () async {
      if (!mounted) return;
      await _ctrl.reverse();
      if (mounted) widget.onRequestClose();
    });
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      ignoring: false,
      child: Container(
        width: double.infinity,
        alignment: Alignment.topCenter,
        child: SlideTransition(
          position: _offset,
          child: FadeTransition(opacity: _fade, child: widget.child),
        ),
      ),
    );
  }
}
