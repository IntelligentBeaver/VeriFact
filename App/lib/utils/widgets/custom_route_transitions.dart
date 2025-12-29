import 'package:animations/animations.dart';
import 'package:flutter/material.dart';
import 'package:verifact_app/utils/constants/app_globals.dart';
import 'package:verifact_app/extensions/context_extensions.dart';

/// Creates the Custom FadeForwards Page Transitions with modified animations durations and curves.
class CustomFadeForwardsRoute<T> extends MaterialPageRoute<T> {
  /// Constructor for Custom Fade Forwards Route
  CustomFadeForwardsRoute({
    required super.builder,
    super.settings,
  });

  @override
  Duration get transitionDuration => const Duration(milliseconds: 600);

  @override
  Duration get reverseTransitionDuration => const Duration(milliseconds: 300);

  @override
  Widget buildTransitions(
    BuildContext context,
    Animation<double> animation,
    Animation<double> secondaryAnimation,
    Widget child,
  ) {
    return FadeForwardsPageTransitionsBuilder(
      backgroundColor: context.theme.scaffoldBackgroundColor,
    ).buildTransitions(
      this,
      context,
      // animation,
      CurvedAnimation(parent: animation, curve: Curves.decelerate),
      CurvedAnimation(parent: secondaryAnimation, curve: Curves.decelerate),
      // secondaryAnimation,
      child,
    );
  }
}

/// Created a Custom FadeThrough Transition with modified animations durations and curves.
class CustomFadeThroughTransition extends MaterialPageRoute<Widget> {
  /// Creates a Fade Transition between Screens
  CustomFadeThroughTransition({
    required super.builder,
    super.settings,
  });

  @override
  Duration get transitionDuration => const Duration(milliseconds: 400);

  @override
  Widget buildTransitions(
    BuildContext context,
    Animation<double> animation,
    Animation<double> secondaryAnimation,
    Widget child,
  ) {
    return FadeThroughPageTransitionsBuilder(
      fillColor:
          scaffoldMessengerKey.currentContext!.theme.scaffoldBackgroundColor,
    ).buildTransitions(
      this,
      context,
      // animation,
      CurvedAnimation(parent: animation, curve: Curves.ease),
      secondaryAnimation,
      child,
    );
  }
}

class CustomPredictiveBackTransition extends MaterialPageRoute<Widget> {
  /// Creates a Fade Transition between Screens
  CustomPredictiveBackTransition({
    required super.builder,
    super.settings,
  });

  @override
  Duration get transitionDuration => const Duration(milliseconds: 400);

  @override
  Duration get reverseTransitionDuration => const Duration(milliseconds: 300);

  @override
  Widget buildTransitions(
    BuildContext context,
    Animation<double> animation,
    Animation<double> secondaryAnimation,
    Widget child,
  ) {
    return const PredictiveBackPageTransitionsBuilder().buildTransitions(
      this,
      context,
      CurvedAnimation(
        parent: animation,
        curve: Curves.decelerate,
        reverseCurve: Curves.decelerate,
      ),
      secondaryAnimation,
      child,
    );
  }
}

class CustomPredictiveBackTransitionBuilder extends PageTransitionsBuilder {
  @override
  Duration get transitionDuration => const Duration(milliseconds: 400);

  @override
  Duration get reverseTransitionDuration => const Duration(milliseconds: 300);

  @override
  Widget buildTransitions<T>(
    PageRoute<T> route,
    BuildContext context,
    Animation<double> animation,
    Animation<double> secondaryAnimation,
    Widget child,
  ) {
    return const PredictiveBackPageTransitionsBuilder().buildTransitions(
      route,
      context,
      CurvedAnimation(parent: animation, curve: Curves.decelerate),
      CurvedAnimation(parent: secondaryAnimation, curve: Curves.easeInQuart),
      child,
    );
  }
}

class CustomCupertinoPageTransitionBuilder extends PageTransitionsBuilder {
  @override
  Duration get transitionDuration => const Duration(milliseconds: 300);

  @override
  Duration get reverseTransitionDuration => const Duration(milliseconds: 200);

  @override
  Widget buildTransitions<T>(
    PageRoute<T> route,
    BuildContext context,
    Animation<double> animation,
    Animation<double> secondaryAnimation,
    Widget child,
  ) {
    return const CupertinoPageTransitionsBuilder().buildTransitions(
      route,
      context,
      CurvedAnimation(parent: animation, curve: Curves.decelerate),
      CurvedAnimation(parent: secondaryAnimation, curve: Curves.easeInQuart),
      child,
    );
  }
}

class CustomCupertinoPageTransition extends MaterialPageRoute<Widget> {
  /// Creates a Cupertino Transition between Screens
  CustomCupertinoPageTransition({
    required super.builder,
    super.settings,
  });

  @override
  Duration get transitionDuration => const Duration(milliseconds: 400);

  @override
  Duration get reverseTransitionDuration => const Duration(milliseconds: 200);

  @override
  Widget buildTransitions(
    BuildContext context,
    Animation<double> animation,
    Animation<double> secondaryAnimation,
    Widget child,
  ) {
    return const CupertinoPageTransitionsBuilder().buildTransitions(
      this,
      context,
      animation,
      // CurvedAnimation(parent: animation, curve: Curves.ease),
      secondaryAnimation,
      // CurvedAnimation(parent: secondaryAnimation, curve: Curves.ease),
      child,
    );
  }
}
