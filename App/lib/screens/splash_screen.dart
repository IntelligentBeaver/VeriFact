import 'dart:async';

import 'package:flutter/material.dart';
import 'package:verifact_app/screens/home_screen.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with SingleTickerProviderStateMixin {
  double _opacity = 1;

  @override
  void initState() {
    super.initState();

    // Show the splash for a short moment then fade out and navigate.
    // Total visible time: ~900ms then fade for 350ms.
    Timer(const Duration(milliseconds: 1300), () {
      setState(() => _opacity = 0.0);

      // After fade completes, navigate to HomeScreen with a fade transition.
      Timer(const Duration(milliseconds: 350), () {
        Navigator.of(context).pushReplacement(
          PageRouteBuilder<void>(
            pageBuilder: (context, animation, secondaryAnimation) =>
                const HomeScreen(),
            transitionsBuilder:
                (context, animation, secondaryAnimation, child) {
                  return FadeTransition(opacity: animation, child: child);
                },
            transitionDuration: const Duration(milliseconds: 400),
          ),
        );
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: AnimatedOpacity(
          opacity: _opacity,
          duration: const Duration(milliseconds: 350),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // The PNG asset should be added at assets/logos/verifact_badge.png
              Image.asset(
                'assets/icons/verified.png',
                width: 140,
                height: 140,
                fit: BoxFit.contain,
              ),
              const SizedBox(height: 18),
              const Text(
                'Verifact',
                style: TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                  color: Color(0xFF0A6ED1),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
