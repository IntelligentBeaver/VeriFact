import 'package:flutter/animation.dart';
import 'package:flutter/foundation.dart';

class Getters {
  Getters._();

  static Future<dynamic> get delay async {
    if (kDebugMode) return Future.delayed(const Duration(seconds: 3));
  }

  static Duration get duration200 {
    return const Duration(milliseconds: 200);
  }

  static Duration get duration250 {
    return const Duration(milliseconds: 250);
  }

  static Curve get decelerate {
    return Curves.decelerate;
  }

  static Curve get easeInQuart {
    return Curves.easeInQuart;
  }
}
