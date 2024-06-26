#include <iostream>
#include <string>

class AudioSegment {
public:
    // Constructors
    AudioSegment(double duration_seconds) 
        : duration_seconds(duration_seconds), audio_filename(""), subtitle_text("") {}

    AudioSegment(const std::string& audio_filename, const std::string& subtitle_text)
        : duration_seconds(0), audio_filename(audio_filename), subtitle_text(subtitle_text) {}

    AudioSegment(const std::string& subtitle_text)
        : duration_seconds(0), audio_filename("no_audio_file_provided"), subtitle_text(subtitle_text) {}

    // Function to check if the audio segment represents silence
    bool is_silence() const {
        return audio_filename.empty() && subtitle_text.empty() && duration_seconds > 0;
    }

    // Getter methods
    double get_duration_seconds() const {
        return duration_seconds;
    }

    std::string get_audio_filename() const {
        return audio_filename;
    }

    std::string get_subtitle_text() const {
        return subtitle_text;
    }

    void display() const {
        std::cout << "Duration (seconds): " << duration_seconds << std::endl;
        std::cout << "Audio Filename: " << audio_filename << std::endl;
        std::cout << "Subtitle Text: " << subtitle_text << std::endl;
    }

private:
    double duration_seconds;
    std::string audio_filename;
    std::string subtitle_text;
};
