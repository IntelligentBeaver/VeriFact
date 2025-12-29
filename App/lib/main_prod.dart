import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:verifact_app/main_common.dart';
import 'package:verifact_app/utils/constants/enums.dart';

Future<void> main() async {
  try {
    await dotenv.load();
  } catch (e) {
    debugPrint('Error loading .env file: $e');
  }

  // Validate required environment variables for production
  final baseUrl = dotenv.env['API_BASE_URL_PROD'];
  if (baseUrl == null || baseUrl.isEmpty) {
    throw Exception(
      'FATAL: API_BASE_URL_PROD not found in .env file. '
      'Production build requires valid API configuration.',
    );
  }

  await mainCommon(
    flavor: Flavor.prod,
    baseUrl: baseUrl,
    name: 'Production',
  );
}
