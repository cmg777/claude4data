# Data

Datasets for Bolivia's sustainable development analysis (339 municipalities).

## Datasets

| Directory | File | Rows | Columns | Description |
| --------- | ---- | ---- | ------- | ----------- |
| regionNames/ | regionNames.csv | 339 | 8 | Municipality identifiers and administrative metadata |
| sdg/ | sdg.csv | 339 | 17 | SDG composite indices (IMDS and index_sdg1-17) |
| sdgVariables/ | sdgVariables.csv | 339 | 65 | Detailed SDG indicators |
| ntl/ | ln_NTLpc.csv | 339 | 19 | Night-time lights per capita (2012-2020) |
| rawData/ | - | - | - | Raw input data (placeholder) |

## Join Key

All datasets can be joined using the `asdf_id` column.

```python
# Python
df = regions.merge(sdg, on='asdf_id').merge(ntl, on='asdf_id')
```

```r
# R
df <- regions %>%
  left_join(sdg, by = 'asdf_id') %>%
  left_join(ntl, by = 'asdf_id')
```

## Key Variables

### regionNames

- `asdf_id` - Primary key for joining
- `mun` - Municipality name
- `dep` - Department name
- `mun_id`, `dep_id` - Numeric identifiers

### sdg

- `imds` - Municipal Sustainable Development Index (composite score)
- `index_sdg1` through `index_sdg17` - Individual SDG scores (0-100 scale)

### ntl

- `ln_NTLpc2012-2020` - Log night-time lights per capita (raw)
- `ln_t400NTLpc2012-2020` - Hodrick-Prescott filtered trend (lambda=400)

## Important

**DO NOT DELETE** any data files. See `CLAUDE.md` for project rules.
