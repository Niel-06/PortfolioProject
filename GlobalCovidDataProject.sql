

---- Skills Used : Aggregate Functions, Joins, CTE (Common Table Expression), Converting data types, Windows Function, Temp Table, Create view


---- Check data for 2 table about covid

Select * 
from CovidInfected

Select *
from CovidDeathAndVaccination

--- Show total cases per location which order by location and date

Select Location, date,  new_cases, total_cases, Population
from CovidInfected
order by 1,2

--- Showing death percentage per location

Select Ci.Location, CI.date, Total_cases, total_deaths, (cast(Total_deaths as int)/total_cases)*100 as Death_Percentage, CI.Population
from CovidInfected CI
Join CovidDeathAndVaccination CD
	on CI.location = CD.location
	and CI.date = CD.date
where CI.continent is not null
order by 1, 2

--- Show what percentage of total cases infected vs population

Select location, date, Population, Total_Cases, (total_cases/population)*100 as Infected_Percentage
from CovidInfected
order by 1, 2

--- Show countries with highest infection rate compare to population

Select location, population, Max(total_cases) as HighestInfectionRate, Max((total_cases/population))*100 as Infected_Percentage
from CovidInfected
group by location, population
order by Infected_Percentage desc

--- Show Global number of death percentage per continent

Select CInfected.continent, Sum(new_cases) as TotalCases, Sum(Cast(new_deaths as int)) as TotalDeath, (Sum(cast(new_deaths as int))/Sum(New_Cases))*100 as DeathPercentage
from CovidInfected CInfected
join CovidDeathAndVaccination CDeath
	on CInfected.continent = CDeath.continent
	and CInfected.date = CDeath.date
Where CInfected.continent is not null
group by CInfected.continent
order by DeathPercentage desc


--- Show running numbers of vaccinated per day and location

Select CInfected.Continent, CInfected.location, CInfected.date, CInfected.population, CDeath.new_vaccinations,
SUM(Cast(CDeath.new_vaccinations as int)) over (partition by CInfected.location order by CInfected.date) as RunningVaccinatedCount
from CovidInfected CInfected
join CovidDeathAndVaccination CDeath
	on CInfected.location = CDeath.location
	and CInfected.date = CDeath.date
Where CInfected.continent is not null
order by 2,3

--- USING CTE (Common table Expression)

With PopulationVsVaccination (Continent, location, date, population, new_vaccinations,RunningVaccinatedCount) as
(
Select CInfected.Continent, CInfected.location, CInfected.date, CInfected.population, CDeath.new_vaccinations,
SUM(Cast(CDeath.new_vaccinations as int)) over (partition by CInfected.location order by CInfected.date) as RunningVaccinatedCount
from CovidInfected CInfected
join CovidDeathAndVaccination CDeath
	on CInfected.location = CDeath.location
	and CInfected.date = CDeath.date
Where CInfected.continent is not null
)

Select *, (RunningVaccinatedCount/population)*100 as PercentageofVaccinated
from PopulationVsVaccination

--- USING TempTable instead 

Drop table if exists PercentageofPeopleVaccinated
Create table PercentageofPeopleVaccinated
(
Continent nvarchar(255),
location nvarchar(255),
Date Datetime,
Population numeric,
New_vaccination numeric,
RunningVaccinatedCount numeric
)

Insert into PercentageofPeopleVaccinated
Select CInfected.Continent, CInfected.location, CInfected.date, CInfected.population, CDeath.new_vaccinations,
SUM(Cast(CDeath.new_vaccinations as int)) over (partition by CInfected.location order by CInfected.date) as RunningVaccinatedCount
from CovidInfected CInfected
join CovidDeathAndVaccination CDeath
	on CInfected.location = CDeath.location
	and CInfected.date = CDeath.date
Where CInfected.continent is not null

Select *, (RunningVaccinatedCount/Population) as PercentageofVaccinated
from PercentageofPeopleVaccinated

--- Creating view

Create View GlobalNumbers as
Select Location, date,  new_cases, total_cases, Population
from CovidInfected

Create View DeathPercentage as 
Select Ci.Location, CI.date, Total_cases, total_deaths, (cast(Total_deaths as int)/total_cases)*100 as Death_Percentage, CI.Population
from CovidInfected CI
Join CovidDeathAndVaccination CD
	on CI.location = CD.location
	and CI.date = CD.date
where CI.continent is not null
