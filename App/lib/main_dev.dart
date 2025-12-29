import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:verifact_app/main_common.dart';
import 'package:verifact_app/utils/constants/enums.dart';

Future<void> main() async {
  try {
    try {
      await dotenv.load();
    } catch (e) {
      debugPrint('Error loading .env file: $e');
      // Continue anyway for dev mode
    }

    // Validate required environment variables
    final baseUrl = dotenv.env['API_BASE_URL_DEV'];
    if (baseUrl == null || baseUrl.isEmpty) {
      throw Exception(
        'FATAL: API_BASE_URL_DEV not found in .env file. '
        'Please create a .env file with required variables.',
      );
    }

    await mainCommon(
      flavor: Flavor.dev,
      baseUrl: baseUrl,
      name: 'Development',
    );
  } catch (e, stackTrace) {
    debugPrint('Error in main: $e');
    debugPrint('Stack trace: $stackTrace');
    rethrow;
  }
}
