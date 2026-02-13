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
  /// Returns parsed `QAModel` or throws `ApiException` to bubble to UI.
  Future<QAModel> fetchAnswer(
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

    AppLogger.api('POST ${UrlStrings.qaAnswer}');
    AppLogger.state('Request body: ${jsonEncode(body)}');

    try {
      final Response<dynamic> response = await dio.post<dynamic>(
        UrlStrings.qaAnswer,
        data: body,
      );

      AppLogger.api('Response status: ${response.statusCode}');

      // Success path: parse JSON into QAModel
      if (response.statusCode != null &&
          response.statusCode! >= 200 &&
          response.statusCode! < 300) {
        final data = response.data;
        if (data is Map<String, dynamic>) return QAModel.fromJson(data);
        if (data is String) {
          final decoded = jsonDecode(data);
          if (decoded is Map<String, dynamic>) return QAModel.fromJson(decoded);
        }
        throw ApiException(
          'Unexpected response format from server',
          statusCode: response.statusCode,
        );
      }

      // Non-success HTTP status: extract a useful message if possible
      try {
        final d = response.data;
        if (d is Map && d['message'] != null) {
          throw ApiException(
            d['message'].toString(),
            statusCode: response.statusCode,
          );
        }
        if (d is String && d.trim().isNotEmpty) {
          throw ApiException(d.trim(), statusCode: response.statusCode);
        }
      } catch (_) {}

      throw ApiException(
        'HTTP ${response.statusCode}',
        statusCode: response.statusCode,
      );
    } on DioException catch (e) {
      AppLogger.error('Network error while posting QA', e, e.stackTrace);

      // Map DioException into ApiException inline for clarity
      final status = e.response?.statusCode;
      if (e.type == DioExceptionType.connectionTimeout ||
          e.type == DioExceptionType.sendTimeout ||
          e.type == DioExceptionType.receiveTimeout) {
        return Future.error(
          ApiException('Request timed out. Check your connection.'),
        );
      }

      final resp = e.response;
      if (resp != null) {
        final d = resp.data;
        if (d is Map && d.containsKey('message')) {
          return Future.error(
            ApiException(d['message'].toString(), statusCode: status),
          );
        }
        if (d is String && d.trim().isNotEmpty) {
          return Future.error(ApiException(d.trim(), statusCode: status));
        }
        return Future.error(ApiException('HTTP $status', statusCode: status));
      }

      return Future.error(ApiException(e.message ?? 'Network error'));
    } catch (e, st) {
      AppLogger.error('Unknown error while posting QA', e, st);
      return Future.error(ApiException(e.toString()));
    }
  }
}
