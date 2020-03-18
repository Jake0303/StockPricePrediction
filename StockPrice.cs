// <Snippet1>
using Microsoft.ML.Data;
// </Snippet1>

namespace StockPricePrediction
{
    // <Snippet2>
    public class StockPrice
    {
        [LoadColumn(0)]
        public float Open;

        [LoadColumn(1)]
        public float High;

        [LoadColumn(2)]
        public float Low;

        [LoadColumn(3)]
        public float Close;

        [LoadColumn(4)]
        public float Volume;
    }

    public class StockPriceClosePrediction
    {
        [ColumnName("Score")]
        public float Close;
    }
    // </Snippet2>
}