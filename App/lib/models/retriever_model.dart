// Models for retriever responses

class RetrieverResponse {
  final String? query;
  final int? count;
  final List<RetrieverModel> results;

  RetrieverResponse({required this.results, this.query, this.count});

  factory RetrieverResponse.fromJson(Map<String, dynamic> json) {
    return RetrieverResponse(
      query: json['query'] as String?,
      count: json['count'] is int
          ? json['count'] as int
          : (json['count'] != null
                ? int.tryParse(json['count'].toString())
                : null),
      results:
          (json['results'] as List<dynamic>?)
              ?.map(
                (e) => RetrieverModel.fromJson(
                  Map<String, dynamic>.from(e as Map),
                ),
              )
              .toList() ??
          [],
    );
  }

  Map<String, dynamic> toJson() => {
    'query': query,
    'count': count,
    'results': results.map((r) => r.toJson()).toList(),
  };
}

class RetrieverModel {
  final Passage? passage;
  final double? rrfScore;
  final double? faissScore;
  final double? esScore;
  final int? faissRank;
  final int? esRank;
  final double? crossScore;
  final double? finalScore;
  final String? domainTier;
  final Map<String, dynamic>? scores;

  RetrieverModel({
    this.passage,
    this.rrfScore,
    this.faissScore,
    this.esScore,
    this.faissRank,
    this.esRank,
    this.crossScore,
    this.finalScore,
    this.domainTier,
    this.scores,
  });

  factory RetrieverModel.fromJson(Map<String, dynamic> json) {
    double? toDouble(dynamic v) {
      if (v == null) return null;
      if (v is double) return v;
      if (v is int) return v.toDouble();
      return double.tryParse(v.toString());
    }

    int? toInt(dynamic v) {
      if (v == null) return null;
      if (v is int) return v;
      return int.tryParse(v.toString());
    }

    return RetrieverModel(
      passage: json['passage'] != null
          ? Passage.fromJson(Map<String, dynamic>.from(json['passage'] as Map))
          : null,
      rrfScore: toDouble(json['rrf_score'] ?? json['rrfScore']),
      faissScore: toDouble(json['faiss_score'] ?? json['faissScore']),
      esScore: toDouble(json['es_score'] ?? json['esScore']),
      faissRank: toInt(json['faiss_rank'] ?? json['faissRank']),
      esRank: toInt(json['es_rank'] ?? json['esRank']),
      crossScore: toDouble(json['cross_score'] ?? json['crossScore']),
      finalScore: toDouble(json['final_score'] ?? json['finalScore']),
      domainTier:
          json['domain_tier'] as String? ?? json['domainTier'] as String?,
      scores: json['scores'] != null
          ? Map<String, dynamic>.from(json['scores'] as Map)
          : null,
    );
  }

  Map<String, dynamic> toJson() => {
    'passage': passage?.toJson(),
    'rrf_score': rrfScore,
    'faiss_score': faissScore,
    'es_score': esScore,
    'faiss_rank': faissRank,
    'es_rank': esRank,
    'cross_score': crossScore,
    'final_score': finalScore,
    'domain_tier': domainTier,
    'scores': scores,
  };
}

class Passage {
  final String? passageId;
  final String? docId;
  final String? sectionHeading;
  final int? blockIndex;
  final String? text;
  final String? url;
  final String? title;
  final String? publishedDate;
  final String? scrapeTimestampUtc;
  final String? author;
  final String? medicallyReviewedBy;
  final List<String>? sources;
  final dynamic location;
  final List<String>? tags;
  final String? domainTier;

  Passage({
    this.passageId,
    this.docId,
    this.sectionHeading,
    this.blockIndex,
    this.text,
    this.url,
    this.title,
    this.publishedDate,
    this.scrapeTimestampUtc,
    this.author,
    this.medicallyReviewedBy,
    this.sources,
    this.location,
    this.tags,
    this.domainTier,
  });

  factory Passage.fromJson(Map<String, dynamic> json) {
    return Passage(
      passageId: json['passage_id'] as String? ?? json['passageId'] as String?,
      docId: json['doc_id'] as String? ?? json['docId'] as String?,
      sectionHeading:
          json['section_heading'] as String? ??
          json['sectionHeading'] as String?,
      blockIndex: json['block_index'] is int
          ? json['block_index'] as int
          : (json['block_index'] != null
                ? int.tryParse(json['block_index'].toString())
                : null),
      text: json['text'] as String?,
      url: json['url'] as String?,
      title: json['title'] as String?,
      publishedDate:
          json['published_date'] as String? ?? json['publishedDate'] as String?,
      scrapeTimestampUtc:
          json['scrape_timestamp_utc'] as String? ??
          json['scrapeTimestampUtc'] as String?,
      author: json['author'] as String?,
      medicallyReviewedBy:
          json['medically_reviewed_by'] as String? ??
          json['medicallyReviewedBy'] as String?,
      sources: (json['sources'] as List<dynamic>?)
          ?.map((s) => s.toString())
          .toList(),
      location: json['location'],
      tags: (json['tags'] as List<dynamic>?)?.map((s) => s.toString()).toList(),
      domainTier:
          json['domain_tier'] as String? ?? json['domainTier'] as String?,
    );
  }

  Map<String, dynamic> toJson() => {
    'passage_id': passageId,
    'doc_id': docId,
    'section_heading': sectionHeading,
    'block_index': blockIndex,
    'text': text,
    'url': url,
    'title': title,
    'published_date': publishedDate,
    'scrape_timestamp_utc': scrapeTimestampUtc,
    'author': author,
    'medically_reviewed_by': medicallyReviewedBy,
    'sources': sources,
    'location': location,
    'tags': tags,
    'domain_tier': domainTier,
  };
}
