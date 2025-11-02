-- Create database
CREATE DATABASE IF NOT EXISTS gengrui_msc CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE gengrui_msc;

-- Drop table if exists (for development reset)
DROP TABLE IF EXISTS student_behavior;

-- Create table for student learning behavior
CREATE TABLE IF NOT EXISTS student_behavior (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_id VARCHAR(10),
    login_count INT,
    time_spent FLOAT,
    quiz_attempts INT,
    completion_rate FLOAT,
    quiz_score FLOAT,
    progress FLOAT,
    created_at DATETIME
);
