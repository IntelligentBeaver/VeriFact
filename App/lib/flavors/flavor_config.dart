import 'package:verifact_app/utils/constants/enums.dart';

class FlavorConfig {
  /// factory for the FlavorConfig
  factory FlavorConfig({
    required Flavor flavor,
    required String baseUrl,
    required String name,
    // required String appSchemeName,
    // required String appleRedirectUrl,
  }) {
    /// If instance variable is null, call the constructor and fill the values.
    _instance ??= FlavorConfig._(
      flavor: flavor,
      baseUrl: baseUrl,
      name: name,
      // appSchemeName: appSchemeName,
      // appleRedirectUrl: appleRedirectUrl,
    );
    return _instance!;
  }

  /// Private constructor with required attributes
  FlavorConfig._({
    required this.flavor,
    required this.baseUrl,
    required this.name,
    // required this.appSchemeName,
    // required this.appleRedirectUrl,
  });

  // Static private instance (for singleton)
  static FlavorConfig? _instance;

  /// Getter for the `instance` variable
  static FlavorConfig get instance {
    if (_instance == null) {
      throw Exception('Flavor not initialized');
    }
    return _instance!;
  }

  /// Getter for checking if flavor is [Flavor.dev]
  static bool isDev() => instance.flavor == Flavor.dev;

  /// Getter for checking if flavor is [Flavor.prod]
  static bool isProd() => instance.flavor == Flavor.prod;

  final Flavor flavor;
  final String baseUrl;
  final String name;
  // final String appSchemeName;
  // final String appleRedirectUrl;
}
