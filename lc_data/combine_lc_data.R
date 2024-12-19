library(vroom)

# -------------------------------------------------------------------
# Top left corner lat/long coordinates for all grid sections

sst_grid_names <- vroom("/users/3/boyti003/temp/LC/sst_filenames.txt", col_names = TRUE)

# -------------------------------------------------------------------
# Water Levels

water_levels <- vroom("/users/3/boyti003/temp/LC/water_levels.txt", col_names = TRUE)

# -------------------------------------------------------------------
# Ice Cover

lake_erie_ic <- vroom("/users/3/boyti003/temp/LC/lake_erie_ic.txt", col_names = TRUE)
lake_huron_ic <- vroom("/users/3/boyti003/temp/LC/lake_huron_ic.txt", col_names = TRUE)
lake_michigan_ic <- vroom("/users/3/boyti003/temp/LC/lake_michigan_ic.txt", col_names = TRUE)
lake_ontario_ic <- vroom("/users/3/boyti003/temp/LC/lake_ontario_ic.txt", col_names = TRUE)
lake_superior_ic <- vroom("/users/3/boyti003/temp/LC/lake_superior_ic.txt", col_names = TRUE)

# --------------------------------------------------------------------
# Lake Depth

lake_erie_depth <- vroom("/users/3/boyti003/temp/erie_lld.xyz", col_names = FALSE)
lake_huron_depth <- vroom("/users/3/boyti003/temp/huron_lld.xyz", col_names = FALSE)
lake_michigan_depth <- vroom("/users/3/boyti003/temp/michigan_lld.xyz", col_names = FALSE)
lake_ontario_depth <- vroom("/users/3/boyti003/temp/ontario_lld.xyz", col_names = FALSE)
lake_superior_depth <- vroom("/users/3/boyti003/temp/superior_lld.xyz", col_names = FALSE)

colnames(lake_erie_depth) <- c("longitude", "latitude", "lake_depth")
colnames(lake_huron_depth) <- c("longitude", "latitude", "lake_depth")
colnames(lake_michigan_depth) <- c("longitude", "latitude", "lake_depth")
colnames(lake_ontario_depth) <- c("longitude", "latitude", "lake_depth")
colnames(lake_superior_depth) <- c("longitude", "latitude", "lake_depth")

# ====================================================================

# Collect all coordinate pairs for grid subsections
all_lat <- c()
all_long <- c()

for (i in 1:length(1:nrow(sst_grid_names))) {
  
  lat <- as.numeric(sst_grid_names$latitude[i])
  long <- as.numeric(sst_grid_names$longitude[i])
  
  all_lat <- append(all_lat, c(lat, lat, lat, lat - 0.14, lat - 0.14, lat - 0.14, lat - 0.28, lat - 0.28, lat - 0.28))
  all_long <- append(all_long, c(long, long + 0.14, long + 0.28, long, long + 0.14, long + 0.28, long, long + 0.14, long + 0.28))
  
}

# ---------------------------------------------------------------------
# For each subsection grid, take the average lake depth and combine it
# with ice cover / water level data for all timepoints

any_water <- c()
combined_lc_df <- data.frame()

for (i in 1:length(all_lat)) {
  
  lat <- all_lat[i]
  long <- all_long[i]
  
  # Determine which lake the lat/long coordinates belong to
  lake <- ifelse(lat %in% lake_erie_depth$latitude & long %in% lake_erie_depth$longitude, "Erie",
                 ifelse(lat %in% lake_huron_depth$latitude & long %in% lake_huron_depth$longitude, "Huron",
                        ifelse(lat %in% lake_michigan_depth$latitude & long %in% lake_michigan_depth$longitude, "Michigan",
                               ifelse(lat %in% lake_ontario_depth$latitude & long %in% lake_ontario_depth$longitude, "Ontario",
                                      "Superior"))))
  
  lake_depth_df <- get(paste0("lake_", tolower(lake), "_depth"))
  ice_cover_df <- get(paste0("lake_", tolower(lake), "_ic"))
  water_levels_df <- water_levels[, c("date", lake)]
  
  # Get the relevant grid 
  whole_grid <- unlist(lake_depth_df[
    lake_depth_df$longitude >= long & lake_depth_df$longitude <= long + 0.14 &
      lake_depth_df$latitude <= lat & lake_depth_df$latitude >= lat - 0.14,
    "lake_depth"
  ])
  
  # Check if there is any portion of the lake in the grid
  any_water <- c(any_water, ifelse(sum(whole_grid < 0, na.rm = TRUE) > 0, "water", "all_land"))
  
  # Calculate lake depth
  # If NA or no values in specified grid, take average lake depth of the 10 nearest coordinates
  # Otherwise take average of all non-NA lake depth values
  if (length(whole_grid) != 0 && sum(is.na(whole_grid)) != length(whole_grid)) {
    lake_depth <- mean(whole_grid, na.rm = TRUE)
  } else {
    
    lake_depth_df$distance <- sqrt((lake_depth_df$latitude - lat)^2 + (lake_depth_df$longitude - long)^2)
    lake_depth_df <- lake_depth_df[order(lake_depth_df$distance, na.last = NA), ]
    top_10 <- head(lake_depth_df[!is.na(lake_depth_df$lake_depth), ], 10)
    lake_depth <- mean(top_10$lake_depth, na.rm = TRUE)
  }
  
  # All info for current grid across all timepoints
  lake_condition_df <- data.frame(
    Time = water_levels$date,
    longitude = long,
    latitude = lat,
    lake_depth = lake_depth,
    ice_cover = ice_cover_df$ice_cover,
    water_level = unlist(water_levels_df[, lake])
  )
  
  # Add this grid's info to overall df
  combined_lc_df <- rbind(combined_lc_df, lake_condition_df)
}

write.csv(combined_lc_df, "/users/3/boyti003/temp/LC/combined_LC_data.csv")

any_water_df <- data.frame(
  longitude = all_long,
  latitude = all_lat,
  any_water = any_water)
)

write.csv(any_water_df, "/users/3/boyti003/temp/LC/any_water.csv")

sum(is.na(combined_lc_df$lake_depth))
sum(is.na(combined_lc_df$lake_depth))
sum(any_water_df$any_water == "all_land", na.rm = TRUE)