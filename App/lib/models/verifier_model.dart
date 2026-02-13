import 'dart:convert';

class VerifierModel {
  VerifierModel({
    required this.claim,
    required this.verdict,
    required this.confidence,
    required this.scores,
    required this.evidence,
    required this.retrieverCandidates,
  });

  factory VerifierModel.fromJson(Map<String, dynamic> json) {
    return VerifierModel(
      claim: json['claim']?.toString() ?? '',
      verdict: json['verdict']?.toString() ?? '',
      confidence: parseDouble(json['confidence']),
      scores: Scores.fromJson(json['scores'] as Map<String, dynamic>? ?? {}),
      evidence: json['evidence'] != null
          ? Evidence.fromJson(json['evidence'] as Map<String, dynamic>)
          : null,
      retrieverCandidates:
          (json['retriever_candidates'] ?? json['retrieverCandidates'] ?? 0)
              as int,
    );
  }
  final String claim;
  final String verdict;
  final double? confidence;
  final Scores scores;
  final Evidence? evidence;
  final int retrieverCandidates;

  Map<String, dynamic> toJson() => {
    'claim': claim,
    'verdict': verdict,
    'confidence': confidence,
    'scores': scores.toJson(),
    'evidence': evidence?.toJson(),
    'retriever_candidates': retrieverCandidates,
  };

  @override
  String toString() => jsonEncode(toJson());

  static double? parseDouble(dynamic v) {
    if (v == null) return null;
    if (v is double) return v;
    if (v is int) return v.toDouble();
    if (v is String) return double.tryParse(v);
    return null;
  }
}

class Scores {
  Scores({
    required this.neutral,
    required this.refutes,
    required this.supports,
  });
  factory Scores.fromJson(Map<String, dynamic> json) {
    return Scores(
      neutral: _parse(json['neutral']),
      refutes: _parse(json['refutes']),
      supports: _parse(json['supports']),
    );
  }
  final double neutral;
  final double refutes;
  final double supports;

  Map<String, dynamic> toJson() => {
    'neutral': neutral,
    'refutes': refutes,
    'supports': supports,
  };

  static double _parse(dynamic v) {
    if (v == null) return 0.0;
    if (v is double) return v;
    if (v is int) return v.toDouble();
    if (v is String) return double.tryParse(v) ?? 0.0;
    return 0.0;
  }
}

class Evidence {
  final String title;
  final String url;
  final String text;
  final double score;
  final String passageId;

  Evidence({
    required this.title,
    required this.url,
    required this.text,
    required this.score,
    required this.passageId,
  });

  factory Evidence.fromJson(Map<String, dynamic> json) => Evidence(
    title: json['title']?.toString() ?? '',
    url: json['url']?.toString() ?? '',
    text: json['text']?.toString() ?? '',
    score: VerifierModel.parseDouble(json['score']) ?? 0.0,
    passageId:
        json['passage_id']?.toString() ?? json['passageId']?.toString() ?? '',
  );

  Map<String, dynamic> toJson() => {
    'title': title,
    'url': url,
    'text': text,
    'score': score,
    'passage_id': passageId,
  };
}
