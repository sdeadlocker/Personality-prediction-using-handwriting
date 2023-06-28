import os
import streamlit as st
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import extract
import categorize

class PersonalityAnalyzer:
    """
    A class for analyzing handwriting features and predicting personality traits based on them.

    Attributes:
        X_baseline_angle (list): List of baseline angles.
        X_top_margin (list): List of top margins.
        X_letter_size (list): List of letter sizes.
        X_line_spacing (list): List of line spacings.
        X_word_spacing (list): List of word spacings.
        X_pen_pressure (list): List of pen pressures.
        X_slant_angle (list): List of slant angles.
        y_traits (list): List of trait labels.
        y_t1 (list): List of labels for trait 1.
        y_t2 (list): List of labels for trait 2.
        y_t3 (list): List of labels for trait 3.
        y_t4 (list): List of labels for trait 4.
        y_t5 (list): List of labels for trait 5.
        y_t6 (list): List of labels for trait 6.
        y_t7 (list): List of labels for trait 7.
        y_t8 (list): List of labels for trait 8.
        page_ids (list): List of page IDs.
        clf1 (SVC): Classifier for trait 1.
        clf2 (SVC): Classifier for trait 2.
        clf3 (SVC): Classifier for trait 3.
        clf4 (SVC): Classifier for trait 4.
        clf5 (SVC): Classifier for trait 5.
        clf6 (SVC): Classifier for trait 6.
        clf7 (SVC): Classifier for trait 7.
        clf8 (SVC): Classifier for trait 8.
    """
    def __init__(self):
        """
        Initializes an instance of the HandwritingAnalyzer class.

        """
        self.X_baseline_angle = []
        self.X_top_margin = []
        self.X_letter_size = []
        self.X_line_spacing = []
        self.X_word_spacing = []
        self.X_pen_pressure = []
        self.X_slant_angle = []
        self.y_traits = []
        self.y_t1 = []  # Trait 1
        self.y_t2 = []  # Trait 2
        self.y_t3 = []  # Trait 3
        self.y_t4 = []  # Trait 4
        self.y_t5 = []  # Trait 5
        self.y_t6 = []  # Trait 6
        self.y_t7 = []  # Trait 7
        self.y_t8 = []  # Trait 8
        self.page_ids = []
        self.clf1 = None
        self.clf2 = None
        self.clf3 = None
        self.clf4 = None
        self.clf5 = None
        self.clf6 = None
        self.clf7 = None
        self.clf8 = None
        self.file_name = None

    def load_data(self, label_list_file):
        """
        Loads the data from the label list file.

        Args:
            label_list_file (str): The path or name of the label list file.

        Returns:
            None
        """
        if os.path.isfile(label_list_file):
            st.write("Info: label_list found.")
            with open(label_list_file, "r") as labels:
                for line in labels:
                    content = line.split()

                    baseline_angle = float(content[0])
                    self.X_baseline_angle.append(baseline_angle)

                    top_margin = float(content[1])
                    self.X_top_margin.append(top_margin)

                    letter_size = float(content[2])
                    self.X_letter_size.append(letter_size)

                    line_spacing = float(content[3])
                    self.X_line_spacing.append(line_spacing)

                    word_spacing = float(content[4])
                    self.X_word_spacing.append(word_spacing)

                    pen_pressure = float(content[5])
                    self.X_pen_pressure.append(pen_pressure)

                    slant_angle = float(content[6])
                    self.X_slant_angle.append(slant_angle)

                    trait_1 = float(content[7])
                    self.y_t1.append(trait_1)

                    trait_2 = float(content[8])
                    self.y_t2.append(trait_2)

                    trait_3 = float(content[9])
                    self.y_t3.append(trait_3)

                    trait_4 = float(content[10])
                    self.y_t4.append(trait_4)

                    trait_5 = float(content[11])
                    self.y_t5.append(trait_5)

                    trait_6 = float(content[12])
                    self.y_t6.append(trait_6)

                    trait_7 = float(content[13])
                    self.y_t7.append(trait_7)

                    trait_8 = float(content[14])
                    self.y_t8.append(trait_8)

                    page_id = content[15]
                    self.page_ids.append(page_id)
        else:
            st.write("Error: label_list file not found.")

    def train_models(self):
        """
        Trains the SVM models for each personality trait.

        Returns:
            None
        """

        # emotional stability
        X_t1 = []
        for a, b in zip(self.X_baseline_angle, self.X_slant_angle):
            X_t1.append([a, b])

        # mental energy or will power
        X_t2 = []
        for a, b in zip(self.X_letter_size, self.X_pen_pressure):
            X_t2.append([a, b])

        # modesty
        X_t3 = []
        for a, b in zip(self.X_letter_size, self.X_top_margin):
            X_t3.append([a, b])

        # personal harmony and flexibility
        X_t4 = []
        for a, b in zip(self.X_line_spacing, self.X_word_spacing):
            X_t4.append([a, b])

        # lack of discipline
        X_t5 = []
        for a, b in zip(self.X_slant_angle, self.X_top_margin):
            X_t5.append([a, b])

        # poor concentration
        X_t6 = []
        for a, b in zip(self.X_letter_size, self.X_line_spacing):
            X_t6.append([a, b])

        # non communicativeness
        X_t7 = []
        for a, b in zip(self.X_letter_size, self.X_word_spacing):
            X_t7.append([a, b])

        # social isolation
        X_t8 = []
        for a, b in zip(self.X_line_spacing, self.X_word_spacing):
            X_t8.append([a, b])


        X_train, X_test, y_train, y_test = train_test_split(X_t1, 
            self.y_t1, test_size=.30, random_state=8)
        self.clf1 = SVC(kernel='rbf')
        self.clf1.fit(X_train, y_train)

        X_train, X_test, y_train, y_test = train_test_split(
            X_t2, self.y_t2, test_size=.30, random_state=16 )
        self.clf2 = SVC(kernel='rbf')
        self.clf2.fit(X_train, y_train)

        X_train, X_test, y_train, y_test = train_test_split(
            X_t3, self.y_t3, test_size=.30, random_state=32)
        self.clf3 = SVC(kernel='rbf')
        self.clf3.fit(X_train, y_train)

        X_train, X_test, y_train, y_test = train_test_split(
            X_t4, self.y_t4, test_size=.30, random_state=64 )
        self.clf4 = SVC(kernel='rbf')
        self.clf4.fit(X_train, y_train)

        X_train, X_test, y_train, y_test = train_test_split(
           X_t5, self.y_t5, test_size=.30, random_state=42)
        self.clf5 = SVC(kernel='rbf')
        self.clf5.fit(X_train, y_train)

        X_train, X_test, y_train, y_test = train_test_split(
            X_t6, self.y_t6, test_size=.30, random_state=52 )
        self.clf6 = SVC(kernel='rbf')
        self.clf6.fit(X_train, y_train)

        X_train, X_test, y_train, y_test = train_test_split(
            X_t7, self.y_t7, test_size=.30, random_state=21)
        self.clf7 = SVC(kernel='rbf')
        self.clf7.fit(X_train, y_train)

        X_train, X_test, y_train, y_test = train_test_split(
            X_t8, self.y_t8, test_size=.30, random_state=73)
        self.clf8 = SVC(kernel='rbf')
        self.clf8.fit(X_train, y_train)



    def analyze_handwriting(self):
        """
        Analyzes the handwriting features of the given image and predicts the personality traits.

        Args:
            file_name (str): The path or name of the handwriting image file.

        Returns:
            None
        """
        # Create the Streamlit app
        st.title("Personality performance Analysis from Handwriting")
        uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

        # Create the uploads directory if it doesn't exist
        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        if uploaded_file is not None:
            # Perform handwriting analysis
            st.write("Pls wait! Generating results")  

            # Save the uploaded file locally
            file_path = os.path.join("uploads", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            self.file_name = file_path    
            
            # raw_features = extract.start(file_path)
            st.subheader("BELOW ARE HANDWRITING FEATURES")

        if self.file_name is not None:
            raw_features = extract.start(self.file_name)

            raw_baseline_angle = raw_features[0]
            baseline_angle, comment = categorize.determine_baseline_angle( raw_baseline_angle)
            st.write ("Baseline Angle: "+comment)

            raw_top_margin = raw_features[1]
            top_margin, comment = categorize.determine_top_margin(raw_top_margin)
            st.write ("Top Margin: "+comment)

            raw_letter_size = raw_features[2]
            letter_size, comment = categorize.determine_letter_size( raw_letter_size)
            st.write ("Letter Size: "+comment)

            raw_line_spacing = raw_features[3]
            line_spacing, comment = categorize.determine_line_spacing(raw_line_spacing)
            st.write ("Line Spacing: "+comment)

            raw_word_spacing = raw_features[4]
            word_spacing, comment = categorize.determine_word_spacing(raw_word_spacing)
            st.write ("Word Spacing: "+comment)

            raw_pen_pressure = raw_features[5]
            pen_pressure, comment = categorize.determine_pen_pressure(raw_pen_pressure)
            st.write ("Pen Pressure: "+comment)

            raw_slant_angle = raw_features[6]
            slant_angle, comment = categorize.determine_slant_angle(raw_slant_angle)
            st.write ("Slant: "+comment)
            st.write("---------------------------------------------------")

            # Mapping of trait predictions to readable labels
            trait_mapping = {0: "Trait is Not Obseved", 1: "Trait is Observed"}

            st.subheader("BELOW ARE PERSONALITY TRAITS")
            st.write("---------------------------------------------------")
            st.write("Emotional Stability: ", trait_mapping[self.clf1.predict([[baseline_angle, slant_angle]])[0]])
            st.write("Mental Energy or Will Power: ", trait_mapping[self.clf2.predict([[letter_size, pen_pressure]])[0]])
            st.write("Modesty: ", trait_mapping[self.clf3.predict([[letter_size, top_margin]])[0]])
            st.write("Personal Harmony and Flexibility: ", trait_mapping[self.clf4.predict([[line_spacing, word_spacing]])[0]])
            st.write("Lack of Discipline: ", trait_mapping[self.clf5.predict([[slant_angle, top_margin]])[0]])
            st.write("Poor Concentration: ", trait_mapping[self.clf6.predict([[letter_size, line_spacing]])[0]])
            st.write("Non Communicativeness: ", trait_mapping[self.clf7.predict([[letter_size, word_spacing]])[0]])
            st.write("Social Isolation: ", trait_mapping[self.clf8.predict([[line_spacing, word_spacing]])[0]])
            st.write("---------------------------------------------------")


analyzer = PersonalityAnalyzer()
# set path for label list
label_list_file = "C:/Users/Admin/MY WORK/ML-graphology/label_list"
analyzer.load_data(label_list_file)
analyzer.train_models()
analyzer.analyze_handwriting()
