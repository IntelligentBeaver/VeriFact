import 'dart:async';
import 'dart:convert';

import 'package:dio/dio.dart';
import 'package:verifact_app/models/qa_model.dart';
import 'package:verifact_app/utils/constants/url_strings.dart';
import 'package:verifact_app/utils/services/app_loggger_services.dart';
import 'package:verifact_app/utils/services/dio_services.dart';

class ApiException implements Exception {
  final String message;
  final int? statusCode;

  ApiException(this.message, {this.statusCode});

  @override
  String toString() => 'ApiException: $message (${statusCode ?? 'n/a'})';
}

/// Singleton controller for QA endpoints.
class QAController {
  QAController._();
  static final QAController instance = QAController._();

  /// POST /qa/answer
  /// body: {"question": "...", "top_k": 10, "min_score": 0.4}
  /// Returns parsed `QAResponse` or throws `ApiException` to bubble to UI.
  Future<QAResponse> fetchAnswer(
    String question, {
    int topK = 10,
    double minScore = 0.4,
  }) async {
    final dio = await DioClient.initClient();
    final body = {
      'question': question,
      'top_k': topK,
      'min_score': minScore,
    };

    try {
      AppLogger.api('POST ${UrlStrings.qaAnswer}');
      AppLogger.state('Request body: ${jsonEncode(body)}');

      final Response<dynamic> response = await dio.post<dynamic>(
        UrlStrings.qaAnswer,
        data: body,
      );

      AppLogger.api('Response status: ${response.statusCode}');

      if (response.statusCode != null &&
          response.statusCode! >= 200 &&
          response.statusCode! < 300) {
        final data = response.data;
        if (data is Map<String, dynamic>) return QAResponse.fromJson(data);
        if (data is String) {
          final decoded = jsonDecode(data);
          if (decoded is Map<String, dynamic>)
            return QAResponse.fromJson(decoded);
        }
        throw ApiException(
          'Unexpected response format from server',
          statusCode: response.statusCode,
        );
      }

      final message = _extractMessage(response);
      throw ApiException(message, statusCode: response.statusCode);
    } on DioException catch (e) {
      AppLogger.error('Network error while posting QA', e, e.stackTrace);
      throw _mapDioException(e);
    } catch (e, st) {
      AppLogger.error('Unknown error while posting QA', e, st);
      throw ApiException(e.toString());
    }
  }

  ApiException _mapDioException(DioException e) {
    final status = e.response?.statusCode;
    final message = e.response != null
        ? _extractMessage(e.response!)
        : (e.message ?? 'Network error');
    return ApiException(message, statusCode: status);
  }

  String _extractMessage(Response<dynamic> response) {
    try {
      final data = response.data;
      if (data is Map && data['message'] != null)
        return data['message'].toString();
      if (data is String) {
        final trimmed = data.trim();
        if (trimmed.isNotEmpty) return trimmed;
      }
    } catch (_) {}
    return 'HTTP ${response.statusCode}';
  }
}
