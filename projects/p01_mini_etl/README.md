# P01 — Mini-ETL

## What it does

A dataset-agnostic command-line ETL: it loads a messy tabular file, audits it
for missing values, replaces a configurable sentinel with `NaN`, coerces
chosen columns to numeric, imputes per-column (mean / mode / drop), and
exports the result as CSV + JSON + Excel plus a plain-text cleaning report.

```
load_data -> audit (before) -> replace_sentinel -> coerce_numeric
          -> impute -> audit (after) -> export -> write_report
```

The input path/URL, sentinel, header list and per-column strategies are all
supplied on the command line or through a JSON config, so the same tool
cleans any tabular dataset without code changes. It reads CSV, TSV, TXT
(with a custom delimiter), JSON (list-of-records) and Excel, and always
writes CSV + JSON + XLSX.

## Run

```bash
uv run p01-mini-etl --demo                       # shipped synthetic fixture
uv run p01-mini-etl --config configs/penguins_biology.json
uv run p01-mini-etl --input my.csv --sentinel "?" \
    --numeric-cols age,income --mean-cols age --mode-cols city --drop-cols id
uv run p01-mini-etl --input data.json             # format auto-detected
uv run p01-mini-etl --input data.xlsx --stem report
uv run p01-mini-etl --input euro.csv --sep ";"
uv run p01-mini-etl --input headerless.csv --headers-file headers.csv --drop-cols price
```

`--verbose` logs each pipeline step at `INFO`; by default only the summary
prints. On success the exit code is `0`; on a controlled error it prints
`error: ...` to stderr and exits `1`. Every run writes `{stem}.csv`,
`{stem}.json`, `{stem}.xlsx` and `cleaning_report.txt` into `out-dir`
(default `output`, default stem `clean`).

### CLI flags

| Flag | Meaning | Default |
|------|---------|---------|
| `--config PATH` | JSON config file; CLI flags override its keys | — |
| `--input PATH\|URL` | Input file (csv/tsv/txt/json/xlsx) | required¹ |
| `--format FMT` | Force format instead of by extension | by extension |
| `--sep CHAR` | Field delimiter (e.g. `";"`) | `,` (tab for `.tsv`) |
| `--sentinel STR` | Missing-value marker to turn into `NaN` | `?` |
| `--headers-file PATH` | One-line CSV of column names for headerless input | — |
| `--numeric-cols A,B` | Columns to cast to `float` | none |
| `--mean-cols A,B` | Impute with the column mean | none |
| `--mode-cols A,B` | Impute with the mode | none |
| `--drop-cols A,B` | Drop rows where these are `NaN` | none |
| `--out-dir PATH` | Output directory | `output` |
| `--stem NAME` | Base name for output files | `clean` |
| `--demo` | Run the offline demo on the shipped fixture | — |
| `--verbose` | Log pipeline steps at `INFO` | off |

¹ `--input` may instead come from the `input` key of a `--config` file.

## Example output

`uv run p01-mini-etl --demo`:

```
p01-mini-etl: ok
  rows: 14
  missing_before: 5
  missing_after: 0
```

Running every shipped config once (13 datasets, 6 input forms):

```
Domain            Form               Dataset                  Rows  Nulls  Rows'  Nulls'
----------------  -----------------  -----------------------  ----  -----  -----  ------
aviation          TSV                airline_passengers       144   0      144    0
automotive        JSON               cars_automotive          406   14     406    0
economics         Excel              gdp_economics             77   0       77    0
health            JSON               health_spending          274   0      274    0
botany            Excel              iris_botany               150  0      150    0
retail/tech       CSV (?-sentinel)   laptops_sample             15  5       14    0
film/media        JSON               movies_entertainment     250   606    250    0
biology           JSON               penguins_biology         344   18     344    0
demographics      TSV                population_demographics   77  0       77    0
food/hospitality  CSV (;-delimited)  restaurant_tips           244  0      244    0
finance           TXT (|-delimited)  stocks_finance            437  0      437    0
history           CSV                titanic_history           891  181    891    0
climate           CSV                weather_climate           366  0      366    0
```

`Rows`/`Nulls` are measured after sentinel replacement (before imputation);
`Rows'`/`Nulls'` after. Penguins, cars, movies and titanic carry genuine
missing values, so `mean`/`mode`/`drop` all run on real gaps, not just the
synthetic fixture.

## Design notes

- **Functional transforms.** `replace_sentinel` returns a new frame
  (`df.replace(...)`, no `inplace=True`) — forward-compatible with pandas 3.x,
  where several in-place paths on copies are deprecated or removed.
- **Config + CLI merge.** `build_config` starts from a JSON file (if any) and
  layers CLI flags on top, so a config gives the defaults and a flag on the
  command line always wins.
- **`run()` also accepts a bare config-file dict.** Only `input` is required;
  every other key falls back to the same default `build_config` would use, so
  a shipped `configs/*.json` can be loaded and run directly, not only through
  the CLI (see the `test_every_shipped_config_cleans_its_dataset` test).
- **Output contract.** Library functions never `print`; they log through
  `logging.getLogger(__name__)`. `main()` prints at most six summary lines and
  sets the logging level (`WARNING`, or `INFO` with `--verbose`).
- **openpyxl** is required only for the `.xlsx` leg; if it is missing, export
  logs a warning and continues with CSV + JSON.

## Limits

`anti_examples/` ships eleven deliberately-broken inputs, grouped by how
loudly the tool fails:

| Class | What you see | Severity |
|-------|--------------|----------|
| A — clear, controlled | `error: ...` on stderr, exit 1 | low — you know exactly what and where |
| B — raw traceback | Python stack trace, exit 1 | medium — cryptic, but still stops |
| C — silent | nothing; exit 0; "successful" files | high — corrupted data, no warning |

| File | Demonstrates | Class |
|------|---------------|:-:|
| `a1_inconsistent_sentinels.csv` / `a1_corrected.csv` | Mixed missing-value markers (`N/A`, `-`, empty, `missing`) vs. one documented sentinel | A |
| `a2_currency_and_thousands_separator.csv` | Currency symbols / thousands separators in a numeric column | A |
| `a3_european_decimal_comma.csv` | `;`-delimited, comma-decimal CSV read with anglophone defaults | A |
| `a4_no_header.csv` + `a4_wrong_headers.csv` | Header count that does not match the column count | A |
| `a5_fully_empty_column.csv` | `mode` on a 100%-empty column → `KeyError` | B |
| `a6_mean_on_text.csv` + config | `mean` strategy on a text column → `TypeError` | B |
| `a7_nested_json.json` | JSON wrapped in an object instead of a list-of-records | A |
| `a8_encoding_latin1.csv` | Latin-1 file read with the tool's fixed `utf-8` decoder | A |
| `a9_leading_zeros.csv` | Leading-zero codes silently coerced to `int` at read time | C (silent) |

Three robustness gaps these expose (none is a correctness bug in what the
tool does today): `impute` raises a raw `KeyError` for `mode` on an
all-missing column and a raw `TypeError` for `mean` on text instead of a
named `error:`; and there is no `--string-cols` escape hatch, so an
identifier that "looks numeric" (postal code, leading-zero ID) is corrupted
silently at `read_csv` time — the only silent (class C) failure of the set.

## Datasets and licences

Twelve real datasets plus one synthetic fixture, spanning twelve domains and
six input forms (JSON, CSV, `;`-CSV, TSV, Excel, `|`-TXT), each under 100 KB:

| Dataset | Domain | Form | Source & licence |
|---------|--------|------|-------------------|
| `penguins_biology.json` | biology | JSON | Palmer Penguins via vega-datasets — CC0 |
| `cars_automotive.json` | automotive | JSON | UCI *Auto MPG* via vega-datasets — public domain (UCI ML Repository); whitespace-minified to stay under 100 KB |
| `gdp_economics.xlsx` | economics | Excel | World Bank GDP via datahub.io/core/gdp — ODC-BY, filtered 7 countries 2010-2020 |
| `population_demographics.tsv` | demographics | TSV | World Bank population via datahub.io/core/population — ODC-BY, filtered 7 countries 2010-2020 |
| `weather_climate.csv` | climate | CSV | Seattle weather via vega-datasets — public domain (NOAA), filtered to 2012 |
| `movies_entertainment.json` | film/media | JSON | vega-datasets *movies* (first 250 rows) — BSD-3-Clause; rich real missing values |
| `titanic_history.csv` | history | CSV | Titanic passenger list via seaborn-data — public domain; `deck` dropped (77% missing) |
| `restaurant_tips.csv` | food/hospitality | CSV (`;`) | Tips (Bryant & Smith, 1995) via seaborn-data — public domain; semicolon-delimited |
| `airline_passengers.tsv` | aviation | TSV | AirPassengers (Box & Jenkins) via seaborn-data — public domain |
| `iris_botany.xlsx` | botany | Excel | Fisher's *Iris* (1936) via seaborn-data / UCI — public domain |
| `stocks_finance.txt` | finance | TXT (`\|`) | Tech-stock monthly prices via vega-datasets — BSD-3-Clause; filtered to four tickers (MSFT, AMZN, GOOG, AAPL) |
| `health_spending.json` | health | JSON | Health spending vs. life expectancy — Our World in Data (from OECD data), CC BY 4.0, via seaborn-data |
| `data/sample_raw.csv` | retail/tech | CSV (`?`-sentinel) | synthetic, written for this repository |

Datasets 6-12 were fetched from vega-datasets / seaborn-data and
re-serialised into the form listed above to exercise every input type the
tool supports; the underlying values are unchanged from the upstream source.

## Courses drawn on

- 4 Python for Data Science, AI & Development
- 7 Data Analysis with Python
