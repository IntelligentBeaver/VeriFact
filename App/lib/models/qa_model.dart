import 'dart:convert';

class QASource {

  QASource({
    required this.id,
    required this.title,
    required this.url,
    required this.text,
    required this.score,
  });

  factory QASource.fromJson(Map<String, dynamic> json) => QASource(
    id: json['id'] is int
        ? json['id'] as int
        : int.tryParse('${json['id']}') ?? 0,
    title: json['title']?.toString() ?? '',
    url: json['url']?.toString() ?? '',
    text: json['text']?.toString() ?? '',
    score: (json['score'] is num)
        ? (json['score'] as num).toDouble()
        : double.tryParse('${json['score']}') ?? 0.0,
  );
  final int id;
  final String title;
  final String url;
  final String text;
  final double score;

  Map<String, dynamic> toJson() => {
    'id': id,
    'title': title,
    'url': url,
    'text': text,
    'score': score,
  };

  @override
  String toString() => jsonEncode(toJson());
}

class QAModel {

  QAModel({
    required this.question,
    required this.answer,
    required this.sources,
  });

  factory QAModel.fromJson(Map<String, dynamic> json) {
    final rawSources = json['sources'];
    final parsed = <QASource>[];
    if (rawSources is List) {
      for (final item in rawSources) {
        if (item is Map<String, dynamic>) {
          parsed.add(QASource.fromJson(item));
        } else if (item is Map) {
          parsed.add(QASource.fromJson(Map<String, dynamic>.from(item)));
        }
      }
    }

    return QAModel(
      question: json['question']?.toString() ?? '',
      answer: json['answer']?.toString() ?? '',
      sources: parsed,
    );
  }
  final String question;
  final String answer;
  final List<QASource> sources;

  Map<String, dynamic> toJson() => {
    'question': question,
    'answer': answer,
    'sources': sources.map((s) => s.toJson()).toList(),
  };

  QAModel copyWith({
    String? question,
    String? answer,
    List<QASource>? sources,
  }) => QAModel(
    question: question ?? this.question,
    answer: answer ?? this.answer,
    sources: sources ?? this.sources,
  );

  @override
  String toString() => jsonEncode(toJson());
}
