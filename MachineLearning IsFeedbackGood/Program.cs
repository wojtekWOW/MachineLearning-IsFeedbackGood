using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using static Microsoft.ML.Data.DatabaseLoader;

namespace MachineLearning_IsFeedbackGood
{
    class FeedbackTrainingData
    {
        [Column(Ordinal: "0", TypeName ="Label")]
        public bool isGood { get; set; }

        [Column(ordinal: "1")]
        public string feedbackText { get; set; } 

        
    }

    class Program
    {
        static List<FeedbackTrainingData> trainingData =
            new List<FeedbackTrainingData>();
        static List<FeedbackTrainingData> testData =
            new List<FeedbackTrainingData>();

        //Method containing test data
        static void LoadTestData()
        {
            testData.Add(new FeedbackTrainingData()
            {
                feedbackText = "good",
                isGood = true
            });
            testData.Add(new FeedbackTrainingData()
            {
                feedbackText = "bad",
                isGood = false
            });
            testData.Add(new FeedbackTrainingData()
            {
                feedbackText = "sad",
                isGood = false
            });
            testData.Add(new FeedbackTrainingData()
            {
                feedbackText = "happy",
                isGood = true
            });
        }

        //Method containing training data
        static void LoadTrainingData()
        {
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "This is good!",
                isGood= true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "Shitty",
                isGood = false
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "Bad as shit",
                isGood = false
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "Love you",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "I'm so glad",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "I'm happy!",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "It is good day",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "This is happy moment !",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "This is bad!",
                isGood = false
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "This is good",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "This is horrible!",
                isGood = false
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "This is awesome",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "So bad",
                isGood = false
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "Good!",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "Shame on you",
                isGood = false
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "Bless you",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "So bad! You should be ashamed",
                isGood = false
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "You can be proud",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "I'm proud of you",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "Bad bad bad!!!!",
                isGood = false
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "Nice",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "Worst ever",
                isGood = false
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "The best",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "So wrong",
                isGood = false
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "So nice",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "I don't like it",
                isGood = false
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "I will like it",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "Nobody will like it",
                isGood = false
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "Everybody will like it",
                isGood = true
            });
        }

        static void Main(string[] args)
        {
            //Step 1 We need to load training data
            LoadTrainingData();

            //Step 2 Create MLContext object so we can access the features of machine learning
            var mlContext = new MLContext();

            //Step 3 Convert your data in to IDataView
            IDataView dataView = mlContext.Data.LoadFromEnumerable<FeedbackTrainingData>(trainingData);

            //Step 4 Crate the pipeline and define the work flows in it
            var pipeline = mlContext.Transforms.Text.FeaturizeText("FeedbackText", "Features")
                .Append(mlContext.BinaryClassification.Trainers.FastTree
                (numberOfLeaves: 50, numberOfTrees: 50, minimumExampleCountPerLeaf : 1));

            //Step 5 Train the algorithm to crate a model
            var model = pipeline.Fit(dataView);
            //Step 6 Load thetest data and run the test data to check model accuracy
            LoadTestData();
             IDataView dataView1=mlContext.Data.LoadFromEnumerable<FeedbackTrainingData>(trainingData);

            var predictions = model.Transform(dataView1);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine(metrics.Accuracy);
            Console.ReadKey();
        }
    }
}
