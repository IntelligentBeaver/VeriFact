import 'dart:async';

import 'package:flutter/material.dart';
import 'package:skeletonizer/skeletonizer.dart';
import 'package:verifact_app/extensions/context_extensions.dart';
import 'package:verifact_app/utils/constants/app_globals.dart';
import 'package:verifact_app/utils/constants/image_strings.dart';
import 'package:verifact_app/utils/constants/sizes.dart';

/// Renders an avatar image for a given [avatar] URL or asset fallback.
///
/// - If [avatar] is a valid HTTP/HTTPS URL, it loads via [Image.network].
/// - If [avatar] is `null` or not a valid URL, it falls back to
///   [ImageStrings.banner] using [Image.asset].
///
/// The image is rendered with [BoxFit.cover] for consistent sizing.
Widget renderAvatar(String? avatar, BuildContext context) {
  if (avatar != null && avatar.startsWith('http')) {
    return Image.network(
      avatar,
      fit: BoxFit.cover,
    );
  } else {
    return Image.asset(
      avatar ?? ImageStrings.logoPrimary,
      fit: BoxFit.cover,
    );
  }
}

/// Builds a [SwitchAnimationConfig] with customizable animation parameters.
///
/// Parameters:
/// - [duration] - Animation duration (default: 300ms).
/// - [inCurve] - Curve used for switch-in animation (default: [Curves.decelerate]).
/// - [outCurve] - Curve used for switch-out animation (default: [Curves.decelerate]).
///
/// Returns a preconfigured [SwitchAnimationConfig] for consistent animation
/// transitions across the app.
SwitchAnimationConfig buildSwitchAnimationConfig({
  Duration duration = const Duration(milliseconds: 300),
  Curve inCurve = Curves.decelerate,
  Curve outCurve = Curves.decelerate,
}) {
  return SwitchAnimationConfig(
    duration: duration,
    reverseDuration: duration,
    switchInCurve: inCurve,
    switchOutCurve: outCurve,
  );
}

enum BackButtonType {
  general,
  video,
}

Widget renderBackButton(
  BuildContext context,
  void Function()? onTap, {
  BackButtonType buttonType = BackButtonType.general,
}) {
  return Padding(
    padding: EdgeInsets.all(AppSizes.md),
    child: GestureDetector(
      onTap: onTap,
      child: Container(
        padding: EdgeInsets.all(AppSizes.sm),
        decoration: BoxDecoration(
          color: buttonType == BackButtonType.video
              ? context.color.inverseSurface
              : context.color.surface,
          boxShadow: buttonType == BackButtonType.video
              ? []
              : [
                  BoxShadow(
                    spreadRadius: 1,
                    blurRadius: 6,
                    color: Colors.grey.shade300,
                  ),
                ],
          shape: BoxShape.circle,
        ),
        clipBehavior: Clip.hardEdge,
        child: Icon(
          Icons.arrow_back_ios_new_rounded,
          color: buttonType == BackButtonType.video
              ? context.color.surface
              : context.color.inverseSurface,
          size: AppSizes.iconMd,
        ),
      ),
    ),
  );
}

ShimmerEffect renderCustomShimmer() {
  return const ShimmerEffect(
    begin: AlignmentGeometry.centerLeft,
    end: AlignmentGeometry.centerRight,
  );
}

SwitchAnimationConfig renderCustomSwitchAnimation() {
  return const SwitchAnimationConfig(
    duration: Duration(milliseconds: 350),
    reverseDuration: Duration(milliseconds: 200),
    switchOutCurve: Curves.easeInQuart,
    switchInCurve: Curves.easeInQuart,
  );
}

Container renderImageContainer({double scale = 1, double size = 110}) {
  return Container(
    height: size,
    alignment: AlignmentGeometry.center,
    child: Transform.scale(
      scale: scale,
      child: Image.asset(
        ImageStrings.logoPrimary,
        fit: BoxFit.contain,
      ),
    ),
  );
}

/// Shows a bottom sheet that correctly moves up with the keyboard.
/// - callerContext: calling widget context (fallback if navigatorKey context isn't available)
/// - builder: builds the sheet content using the sheetContext
/// - isScrollControlled: defaults to true so sheet resizes with keyboard
Future<T?> renderCustomBottomSheet<T>({
  required BuildContext callerContext,
  required Widget Function(BuildContext sheetContext) builder,
  bool isScrollControlled = true,
  Color? barrierColor,
  bool autoHandleInsets =
      true, // automatically wrap content with AnimatedPadding + SingleChildScrollView
}) {
  final ctx = navigatorKey.currentContext ?? callerContext;

  final completer = Completer<T?>();
  WidgetsBinding.instance.addPostFrameCallback((_) async {
    try {
      final result = await showModalBottomSheet<T>(
        context: ctx,
        isScrollControlled: isScrollControlled,
        backgroundColor: Colors.transparent,
        barrierColor: barrierColor ?? Colors.black54,
        shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
        ),
        clipBehavior: Clip.antiAlias,
        builder: (sheetContext) {
          var content = builder(sheetContext);

          if (autoHandleInsets) {
            // Ensure sheet responds to keyboard and can scroll
            content = SafeArea(
              top: false,
              child: AnimatedPadding(
                duration: const Duration(milliseconds: 160),
                padding: EdgeInsets.only(
                  bottom: MediaQuery.of(sheetContext).viewInsets.bottom,
                ),
                // SingleChildScrollView lets focused TextField scroll into view when needed.
                child: SingleChildScrollView(
                  physics: const ClampingScrollPhysics(),
                  child: content,
                ),
              ),
            );
          } else {
            content = SafeArea(top: false, child: content);
          }

          // Outer container gives rounded white background like a native sheet
          return Container(
            decoration: const BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
            ),
            child: content,
          );
        },
      );
      completer.complete(result);
    } catch (e, st) {
      completer.completeError(e, st);
    }
  });

  return completer.future;
}

Future<T?> showCurvedDialog<T>({
  required BuildContext context,
  required Widget child,
  Duration transitionDuration = const Duration(milliseconds: 250),
  Curve curve = Curves.easeOutQuart,
  Curve reverseCurve = Curves.easeOutQuart,
}) {
  return showGeneralDialog(
    context: context,
    transitionDuration: transitionDuration,
    pageBuilder: (context, animation, secondaryAnimation) => child,
    transitionBuilder: (context, animation, secondaryAnimation, child) {
      return Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          ScaleTransition(
            scale:
                Tween<double>(
                  begin: 0.90,
                  end: 1,
                ).animate(
                  CurvedAnimation(
                    parent: animation,
                    curve: curve,
                    reverseCurve: reverseCurve,
                  ),
                ),
            child: FadeTransition(
              opacity: Tween<double>(begin: 0, end: 1).animate(animation),
              child: child,
            ),
          ),
        ],
      );
    },
  );
}
