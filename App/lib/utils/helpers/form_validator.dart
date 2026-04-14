class FormValidator {
  FormValidator._();
  static String? emailValidator(String? email) {
    if (email == null || email.trim().isEmpty) {
      return 'Email is required';
    }
    if (email.length > 100) {
      return 'Email must not be greater than 100 characters';
    }
    final regex = RegExp(r'^[\w\.-]+@[\w\.-]+\.\w+$');
    return regex.hasMatch(email.trim()) ? null : 'Enter a valid email address';
  }

  static String? passwordValidator(String? password) {
    if (password == null || password.isEmpty) {
      return 'Password is required.';
    }
    if (password.length < 8) {
      return 'Password must be at least 8 characters';
    }
    return null;
  }

  static String? confimPasswordValidator(
    String? confirmPassword,
    String? password,
  ) {
    if (confirmPassword == null || confirmPassword.isEmpty) {
      return 'Please confirm your password';
    }
    return confirmPassword == password ? null : 'Passwords do not match.';
  }

  static String? otpNumber(String? value) {
    if (value == null || value.isEmpty) {
      return 'OTP is required';
    }
    if (value.length != 6) {
      return 'OTP must be 6 digits';
    }
    if (!RegExp(r'^[0-9]+$').hasMatch(value)) {
      return 'OTP must contain digits only';
    }
    return null;
  }

  // Username validator
  // - Required
  // - Min length 3
  // - Max length 30
  // - No whitespaces
  static String? usernameValidator(String? username) {
    if (username == null || username.trim().isEmpty) {
      return 'Username is required';
    }
    if (username.length < 3) {
      return 'Username must be at least 3 characters';
    }
    if (username.length > 30) {
      return 'Username must not be greater than 30 characters';
    }
    if (username.contains(' ')) {
      return 'Username must not contain spaces';
    }
    return null;
  }
}
