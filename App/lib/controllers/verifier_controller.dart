import 'dart:async';
import 'dart:convert';

import 'package:dio/dio.dart';
import 'package:verifact_app/models/verifier_model.dart';
import 'package:verifact_app/utils/constants/url_strings.dart';
import 'package:verifact_app/utils/services/dio_services.dart';

/// Controller responsible for calling the verifier backend.
/// Singleton with two public methods: `verifyClaim` and `verifyClaimWithBody`.
class VerifierException implements Exception {
  VerifierException(this.message, {this.statusCode});
  final String message;
  final int? statusCode;

  @override
  String toString() => 'VerifierException(${statusCode ?? 'N/A'}): $message';
}

class VerifierController {
  VerifierController._();
  static final VerifierController instance = VerifierController._();

  static const String _endpoint = UrlStrings.verifierVerify;

  /// Posts a claim string to the verifier endpoint and returns a parsed model.
  /// Throws [VerifierException] on any failure so the UI can show a message.
  Future<VerifierModel> verifyClaim(String claim) async {
    return verifyClaimWithBody({'claim': claim});
  }

  /// Posts an arbitrary body map to the verifier endpoint and returns a parsed model.
  /// This allows callers to send additional fields in the future.
  Future<VerifierModel> verifyClaimWithBody(Map<String, dynamic> body) async {
    final dio = await DioClient.initPublicClient();

    try {
      const url = '${UrlStrings.baseUrl}$_endpoint';
      final response = await dio.post<dynamic>(
        url,
        data: body,
      );

      final status = response.statusCode ?? 0;
      if (status >= 200 && status < 300) {
        final data = response.data;
        if (data == null) {
          throw VerifierException(
            'Empty response from verifier',
            statusCode: status,
          );
        }

        // If the response is a JSON string, try to decode it
        final jsonBody = data is String
            ? (jsonDecodeSafe(data) ?? {})
            : (data as Map<String, dynamic>);

        return VerifierModel.fromJson(jsonBody);
      }

      // Non-success status
      final message =
          _extractMessageFromResponse(response) ??
          'Request failed with status $status';
      throw VerifierException(message, statusCode: status);
    } on DioException catch (e) {
      final status = e.response?.statusCode;
      final message =
          _extractMessageFromDioError(e) ?? e.message ?? 'Network error';
      throw VerifierException(message, statusCode: status);
    } catch (e) {
      throw VerifierException(e.toString());
    }
  }

  String? _extractMessageFromResponse(Response<dynamic> resp) {
    final d = resp.data;
    if (d == null) return null;
    if (d is Map && d['message'] != null) return d['message'].toString();
    return null;
  }

  String? _extractMessageFromDioError(DioException e) {
    final resp = e.response?.data;
    if (resp == null) return null;
    if (resp is Map && resp['message'] != null) {
      return resp['message'].toString();
    }
    if (resp is String) return resp;
    return null;
  }

  Map<String, dynamic>? jsonDecodeSafe(String s) {
    try {
      final parsed = jsonDecode(s);
      if (parsed is Map<String, dynamic>) return parsed;
      return null;
    } catch (_) {
      return null;
    }
  }
}
