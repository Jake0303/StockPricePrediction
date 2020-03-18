using System;
using System.IO;
using Microsoft.ML;

//Attempt to predict stock prices
namespace StockPricePrediction
{
    class Program
    {
        //Training data csv
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "AAPL-Train.csv");
        //Testing data csv
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "AAPL-Test.csv");
        // </Snippet2>

        static void Main(string[] args)
        {
            Console.WriteLine(Environment.CurrentDirectory);

            // <Snippet3>
            MLContext mlContext = new MLContext(seed: 0);
            // </Snippet3>

            // <Snippet5>
            var model = Train(mlContext, _trainDataPath);
            // </Snippet5>

            // <Snippet14>
            Evaluate(mlContext, model);
            // </Snippet14>

            // <Snippet20>
            TestSinglePrediction(mlContext, model);
            // </Snippet20>
        }

        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            // <Snippet6>
            IDataView dataView = mlContext.Data.LoadFromTextFile<StockPrice>(dataPath, hasHeader: true, separatorChar: ',');
            // </Snippet6>

            // <Snippet7>
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "Close")
                    // </Snippet7>
                    // <Snippet8>
                    // </Snippet8>
                    // <Snippet9>
                    .Append(mlContext.Transforms.Concatenate("Features", "Open","High","Low","Volume"))
                    // </Snippet9>
                    // <Snippet10>
                    .Append(mlContext.Regression.Trainers.FastTree());
            // </Snippet10>

            Console.WriteLine("=============== Create and Train the Model ===============");

            // <Snippet11>
            var model = pipeline.Fit(dataView);
            // </Snippet11>

            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();
            // <Snippet12>
            return model;
            // </Snippet12>
        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            // <Snippet15>
            IDataView dataView = mlContext.Data.LoadFromTextFile<StockPrice>(_testDataPath, hasHeader: true, separatorChar: ',');
            // </Snippet15>

            // <Snippet16>
            var predictions = model.Transform(dataView);
            // </Snippet16>
            // <Snippet17>
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");
            // </Snippet17>

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            // <Snippet18>
            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
            // </Snippet18>
            // <Snippet19>
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
            // </Snippet19>
            Console.WriteLine($"*************************************************");
        }

        private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            //Prediction test
            // Create prediction function and make prediction.
            // <Snippet22>
            var predictionFunction = mlContext.Model.CreatePredictionEngine<StockPrice, StockPriceClosePrediction>(model);
            // </Snippet22>
            //Sample: 
            //Date,Open,High,Low,Close,Volume
            //20191101,249.56,255.93,249.16,255.82,29671000
            // <Snippet23>
            var stockPriceSample = new StockPrice()
            {
                Open= 249.56f,
                High = 255.93f,
                Low = 249.16f,
                Close= 0,  // To predict. Actual/Observed = 255.82
                Volume = 29671000
            };
            // </Snippet23>
            // <Snippet24>
            var prediction = predictionFunction.Predict(stockPriceSample);
            // </Snippet24>
            // <Snippet25>
            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted Close Price: {prediction.Close:0.####}, actual close: 255.82");
            Console.WriteLine($"**********************************************************************");
            // </Snippet25>
        }
    }
}