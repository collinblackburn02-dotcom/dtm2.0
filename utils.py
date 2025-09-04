def resolve_col(df, name):
    # Simple resolver: case-insensitive exact match
    for c in df.columns:
        if c.strip().lower() == name.strip().lower():
            return c
    return None
