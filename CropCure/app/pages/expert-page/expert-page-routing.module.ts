import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';

import { ExpertPagePage } from './expert-page.page';

const routes: Routes = [
  {
    path: '',
    component: ExpertPagePage
  }
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule],
})
export class ExpertPagePageRoutingModule {}
