// import 'dart:io' show Platform;

// import 'package:flutter/foundation.dart';
// import 'package:flutter_dotenv/flutter_dotenv.dart';
// import 'package:google_sign_in/google_sign_in.dart';
// import 'package:verifact_app/utils/helpers/helper_functions.dart';

// class GoogleSignInResult {
//   GoogleSignInResult({this.account, this.serverAuthCode});
//   final GoogleSignInAccount? account;
//   final String? serverAuthCode;
// }

// /// Handles Google Sign-In authentication logic.
// /// Supports initialization, sign-in, and additional scope authorization.
// class GoogleAuthService {
//   /// Private constructor to prevent instantiation.
//   GoogleAuthService._();

//   /// Singleton instance for managing Google sign-in.
//   static final GoogleAuthService _instance = GoogleAuthService._();

//   /// Getter that provides access to the singleton instance.
//   static GoogleAuthService get instance => _instance;

//   //* Attirbutes
//   /// Google Sign-In client instance.
//   final GoogleSignIn _googleSignIn = GoogleSignIn.instance;

//   /// Indicates if the service has been initialized.
//   bool _isInitialized = false;

//   //*Getters
//   /// Getter for Sign-In client instance.
//   GoogleSignIn get googleSignIn => _googleSignIn;

//   /// Getter for checkign if service has been initialized.
//   bool get isInitialized => _isInitialized;

//   //* Methods
//   /// Initializes the Google Sign-In client with provided configuration.
//   Future<void> _initialize() async {
//     if (_isInitialized) return;

//     await _googleSignIn.initialize(
//       // Web client ID for server auth code exchange
//       serverClientId: dotenv.env['GOOGLE_SERVER_CLIENT_ID'],
//       // iOS requires explicit clientId when no GoogleService-Info.plist is present
//       clientId: (kIsWeb || !Platform.isIOS)
//           ? null
//           : (dotenv.env['GOOGLE_IOS_CLIENT_ID'] ??
//                 // Fallback to Info.plist GIDClientID if provided via build-time env
//                 null),
//     );

//     _isInitialized = true;
//     debugPrint('✅ GoogleAuthService Initialization completed.');
//   }

//   /// Initializes the Google Sign-In service and triggers user authentication.
//   ///
//   /// Returns a [GoogleSignInAccount] on successful authentication.
//   /// Throws an [GoogleSignInException] or [Exception] if sign-in fails.
//   static Future<GoogleSignInResult> signInWithGoogle() async {
//     final scopes = [
//       'email',
//       'openid',
//       'profile',
//     ];
//     try {
//       //* Initialize and Sign In with google
//       await instance._initialize();
//       final account = await instance._googleSignIn.authenticate(
//         scopeHint: scopes,
//       );
//       debugPrint('✅ GoogleAuthService User authenticated: $account');

//       //* Requesting server authorization code
//       final serverAuthorization = await account.authorizationClient
//           .authorizeServer(scopes);
//       final serverAuthCode = serverAuthorization?.serverAuthCode;

//       if (serverAuthCode == null || serverAuthCode.isEmpty) {
//         // server auth code not available — handle fallback
//         debugPrint(
//           'serverAuthCode is null',
//         );
//       }

//       if (kDebugMode) {
//         debugPrint('✅Server Auth Code: $serverAuthCode');
//         debugPrint('✅Google Account Details:');
//         debugPrint('✅Google ID: ${account.id}');
//         debugPrint('✅Signed in user: ${account.displayName}');
//         debugPrint('✅Email: ${account.email}');
//       }

//       return GoogleSignInResult(
//         serverAuthCode: serverAuthCode,
//         account: account,
//       );
//     } on GoogleSignInException catch (e) {
//       debugPrint(
//         '❌ GoogleSignInException: code=${e.code}, desc=${e.description}',
//       );
//       final result = e.description!.split('] ')[1];
//       showErrorSnackbar(result);
//       rethrow;
//     } catch (genericError) {
//       debugPrint('Unexpected Error: $genericError');
//       rethrow;
//     }
//   }
// }
